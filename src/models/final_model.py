import torch
from torch import nn
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel

from src.models.backbones.parts import DualPath, MLP
from src.models.tabular_transformer import TabularTransformer


class MultiModalFusionModel(nn.Module):
    """Multi-modal fusion model with cross-modal attention"""
    def __init__(self, num_classes: int = 11255, tabular_transformer_path: str = "./saved/tabular_transformer/testing/model_best.pth"):
        super().__init__()

        # Environment variables encoder (5 features)
        self.tabular_transformer = TabularTransformer(d_model=256, nhead=8, num_layers=5, dim_feedforward=256, dropout=0.1)
        weights = torch.load(tabular_transformer_path, map_location="cpu", weights_only=False)
        self.tabular_transformer.load_state_dict(weights.get("state_dict", weights), strict=False)

        for param in self.tabular_transformer.parameters():
            param.requires_grad = False

        self.extra_label_classifier = self.tabular_transformer.head
        self.tabular_transformer.head = nn.Identity()

        # Sentinel-2 encoder (4 channels, 64x64)
        clip_model_name = "openai/clip-vit-base-patch32"
        self.preencoder = nn.Conv2d(4, 3, (1,1))
        self.sentinel_encoder = CLIPModel.from_pretrained(clip_model_name)
        self.sentinel_encoder_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # turn off gradients for CLIP encoder
        for param in self.sentinel_encoder.parameters():
            param.requires_grad = False
        
        # LSTM 1: bioclimatic time series (B, 4, 19, 12) -> (512,)
        # Flatten to (B, 4*19*12) = (B, 912)
        self.lstm1 = DualPath(feature_size=16, time_size=12, num_layers=2)
        self.lstm1_fc = nn.Linear(912, 512)

        # LSTM 2: landsat time series (B, 6, 4, 21) -> (512,)
        # Flatten to (B, 6*4*21) = (B, 504)
        self.lstm2 = DualPath(feature_size=4, time_size=21, num_layers=2)
        self.lstm2_fc = nn.Linear(504, 512)
        
        # Cross-modal attention
        self.pre_attn = nn.LayerNorm(512)
        self.cross_modal_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.post_attn = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512 * 4),
            nn.GELU(),
            nn.Linear(512 * 4, 512)
        )
        
        # Final classifier
        self.output_mlp = MLP(
            input_dim=512 * 3 + 128,
            hidden_dims=[2048],
            output_dim=4096,
            dropout=0.3
        )
        self.head = MLP(
            input_dim=4096,
            hidden_dims=[8192],
            output_dim=num_classes,
            activation=nn.Tanh,
            dropout=0.3
        )

    def forward(self, climate_features, elevation_features, human_footprint_features, 
                land_cover_features, soil_grids_features, satellite, bioclimatic, landsat,
                table_data, **batch):
        """
        Model forward method.

        Args:
            satellite (Tensor): satellite data. (B, 4, 64, 64)
            bioclimatic (Tensor): bioclimatic data. (B, 4, 19, 12)
            landsat (Tensor): landsat data. (B, 6, 4, 21)
            table_data (Tensor): tabular data. (B, 5)
            **batch: other batch data.
        Returns:
            output (dict): output dict containing logits.
        """
        B = satellite.size(0)

        tabular_feat = self.tabular_transformer(
            climate_features, elevation_features, human_footprint_features, 
            land_cover_features, soil_grids_features
        )
        tabular_logits = self.extra_label_classifier(tabular_feat)

        # Encode each modality
        sentinel_inputs = self.sentinel_encoder_processor(images=self.preencoder(satellite), return_tensors="pt").to(satellite.device)
        sentinel_outputs = self.sentinel_encoder(**sentinel_inputs)
        sentinel_feat = sentinel_outputs.pooler_output  # (B, 512)

        bioclim_feat = self.lstm1_fc(self.lstm1(bioclimatic).reshape(B, -1))  # (B, 512)
        landsat_feat = self.lstm2_fc(self.lstm2(landsat).reshape(B, -1))  # (B, 512)

        # Stack features for cross-modal attention
        features = torch.stack([tabular_feat, sentinel_feat, landsat_feat, bioclim_feat], dim=1)  # (B, 3, 512)
        
        # Apply cross-modal attention
        features = self.pre_attn(features)
        features, _ = self.cross_modal_attention(features, features, features)
        features = self.post_attn(features)
        
        # Concatenate all features
        combined_features = features.reshape(features.size(0), -1)  # (B, 3*512)
        
        # Classification
        head_input = self.output_mlp(combined_features)  # (B, 4096)
        logits = self.head(F.relu(head_input))  # (B, num_classes)
        
        return { "logits": logits, "extra_logits": tabular_logits }

import torch
from torch import nn

from src.models.backbones import ResNet
from src.models.backbones.parts import MLP


class ResNetFusionModel(nn.Module):
    """Multi-modal fusion model with cross-modal attention"""
    def __init__(self, num_classes=11254):
        super().__init__()

        # Sentinel-2 encoder (4 channels, 64x64)
        self.sentinel_encoder = ResNet(in_channels=4, hidden_dim=[64, 128, 256], out_channels=512, strides=[2, 2, 2], use_se=True)
        
        # Landsat encoder (6 channels, 4x21)
        self.landsat_encoder = ResNet(in_channels=6, hidden_dim=[64, 128, 256], out_channels=512, strides=[1, 1, 1], use_se=True)
        
        # Bioclim encoder (4 channels, 19x12)
        self.bioclim_encoder = ResNet(in_channels=4, hidden_dim=[64, 128, 256], out_channels=512, strides=[1, 1, 1], use_se=True)
        
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
        self.classifier = MLP(
            512 * 3,
            [1024],
            num_classes
        )

    def forward(self, satellite, bioclimatic, landsat, table_data, **batch):
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
        # Encode each modality
        sentinel_feat = self.sentinel_encoder(satellite)  # (B, 512)
        landsat_feat = self.landsat_encoder(landsat)      # (B, 512)
        bioclim_feat = self.bioclim_encoder(bioclimatic)  # (B, 512)
        
        # Stack features for cross-modal attention
        features = torch.stack([sentinel_feat, landsat_feat, bioclim_feat], dim=1)  # (B, 3, 512)
        
        # Apply cross-modal attention
        features = self.pre_attn(features)
        features, _ = self.cross_modal_attention(features, features, features)
        features = self.post_attn(features)
        
        # Concatenate all features
        combined_features = features.reshape(features.size(0), -1)  # (B, 3*512)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return { "logits": logits }

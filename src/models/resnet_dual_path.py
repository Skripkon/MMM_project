import torch
from torch import nn

from src.models.parts import ResnetBlock, DualPath


class MultiModalFusionModel(nn.Module):
    """Multi-modal fusion model with cross-modal attention"""
    def __init__(self, num_classes=11255):
        super().__init__()

        BasicBlockSE = lambda in_c, out_c, stride=1: ResnetBlock(in_c, out_c, stride=stride, use_se=True)
        
        # Sentinel-2 encoder (4 channels, 64x64)
        self.sentinel_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlockSE(64, 128, 2),  # 32x32
            BasicBlockSE(128, 256, 2),  # 16x16
            BasicBlockSE(256, 512, 2),  # 8x8
            nn.AdaptiveAvgPool2d(1)
        )
        
        # LSTM 1: bioclimatic time series (B, 4, 19, 12) -> (512,)
        # Flatten to (B, 4*19*12) = (B, 912)
        self.lstm1 = DualPath(feature_size=19, time_size=12, num_layers=2)
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
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
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
        B = satellite.size(0)

        # Encode each modality
        sentinel_feat = self.sentinel_encoder(satellite).reshape(B, -1)  # (B, 512)
        bioclim_feat = self.lstm1_fc(self.lstm1(bioclimatic).reshape(B, -1))  # (B, 512)
        landsat_feat = self.lstm2_fc(self.lstm2(landsat).reshape(B, -1))  # (B, 512)
        
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

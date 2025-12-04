import torch
from torch import nn

from src.models.parts import ResnetBlock


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
        
        # Landsat encoder (6 channels, 4x21)
        self.landsat_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlockSE(64, 128),
            BasicBlockSE(128, 256),
            BasicBlockSE(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Bioclim encoder (4 channels, 19x12)
        self.bioclim_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlockSE(64, 128),
            BasicBlockSE(128, 256),
            BasicBlockSE(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )
        
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
        # Encode each modality
        sentinel_feat = self.sentinel_encoder(satellite).squeeze(-1).squeeze(-1)  # (B, 512)
        landsat_feat = self.landsat_encoder(landsat).squeeze(-1).squeeze(-1)    # (B, 512)
        bioclim_feat = self.bioclim_encoder(bioclimatic).squeeze(-1).squeeze(-1)    # (B, 512)
        
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

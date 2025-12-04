import torch
from torch import nn

from src.models.parts import ResnetBlock, DualPath, MLP


class RDPCA(nn.Module):
    """
    Resnet x Dual-Path Cross-Modal Attention model for multi-modal data fusion.
    """
    def __init__(self, num_classes=11255, hidden_dim=512):
        super().__init__()

        BasicBlockSE = lambda in_c, out_c, stride=1: ResnetBlock(in_c, out_c, stride=stride, use_se=True)
        
        # Sentinel-2 encoder (4 channels, 64x64)
        self.sentinel_encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlockSE(64, 128, 2),  # 32x32
            BasicBlockSE(128, 256, 2),  # 16x16
            BasicBlockSE(256, hidden_dim, 2),  # 8x8
            nn.AdaptiveAvgPool2d(1)
        )
        
        # LSTM 1: bioclimatic time series (B, 4, 19, 12) -> (512,)
        # Flatten to (B, 4*19*12) = (B, 912)
        self.lstm1 = DualPath(feature_size=19, time_size=12, num_layers=2)
        self.lstm1_fc = nn.Linear(912, hidden_dim)

        # LSTM 2: landsat time series (B, 6, 4, 21) -> (512,)
        # Flatten to (B, 6*4*21) = (B, 504)
        self.lstm2 = DualPath(feature_size=4, time_size=21, num_layers=2)
        self.lstm2_fc = nn.Linear(504, hidden_dim)

        # MLP: environmental values (B, 5) -> (512,)
        self.mlp_env = MLP(input_dim=5, hidden_dims=[16], output_dim=8, dropout=0.3)

        # Cross-attention modules
        self.sentinel_norm = nn.LayerNorm(hidden_dim)
        self.bioclim_norm = nn.LayerNorm(hidden_dim)
        self.landsat_norm = nn.LayerNorm(hidden_dim)

        self.cross_attn_sentinel_x_bioclim = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.cross_attn_bioclim_x_sentinel = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        self.cross_attn_sentinel_x_landsat = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.cross_attn_landsat_x_sentinel = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        self.post_sentinel_x_bioclim = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.post_bioclim_x_sentinel = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.post_sentinel_x_landsat = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.post_landsat_x_sentinel = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        features_dim = hidden_dim * 4 + 8  # 512*4 + 8 = 2056

        self.pre_attn = nn.LayerNorm(features_dim)
        self.cross_modal_attention = nn.MultiheadAttention(features_dim, num_heads=8, batch_first=True)
        self.post_attn = nn.Sequential(
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim * 4),
            nn.GELU(),
            nn.Linear(features_dim * 4, features_dim)
        )

        # Final MLP: (2048,) -> (num_classes,)
        self.final_mlp = MLP(input_dim=features_dim, hidden_dims=[1024], output_dim=num_classes, dropout=0.3)

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
        table_feat = self.mlp_env(table_data)  # (B, 512)

        sentinel_feat = self.sentinel_norm(sentinel_feat)
        bioclim_feat = self.bioclim_norm(bioclim_feat)
        landsat_feat = self.landsat_norm(landsat_feat)
        
        # Cross-attention: sentinel x bioclim
        sentinel_x_bioclim, _ = self.cross_attn_sentinel_x_bioclim(sentinel_feat, bioclim_feat, bioclim_feat)  # Q=sentinel, K=bioclim, V=bioclim -> (B, 512)
        bioclim_x_sentinel, _ = self.cross_attn_bioclim_x_sentinel(bioclim_feat, sentinel_feat, sentinel_feat)  # Q=bioclim, K=sentinel, V=sentinel -> (B, 512)

        sentinel_x_bioclim = self.post_sentinel_x_bioclim(sentinel_x_bioclim)
        bioclim_x_sentinel = self.post_bioclim_x_sentinel(bioclim_x_sentinel)

        # Cross-attention: sentinel x landsat
        sentinel_x_landsat, _ = self.cross_attn_sentinel_x_landsat(sentinel_feat, landsat_feat, landsat_feat)  # Q=sentinel, K=landsat, V=landsat -> (B, 512)
        landsat_x_sentinel, _ = self.cross_attn_landsat_x_sentinel(landsat_feat, sentinel_feat, sentinel_feat)  # Q=landsat, K=sentinel, V=sentinel -> (B, 512)

        sentinel_x_landsat = self.post_sentinel_x_landsat(sentinel_x_landsat)
        landsat_x_sentinel = self.post_landsat_x_sentinel(landsat_x_sentinel)

        # Concatenate: v1, v2, v3, v4 -> (B, 512 + 512 + 512 + 512 + 8) = (B, 2056)
        features = torch.cat([sentinel_x_bioclim, bioclim_x_sentinel, sentinel_x_landsat, landsat_x_sentinel, table_feat], dim=1)  # (B, 2048)
        
        # Apply cross-modal attention
        features = self.pre_attn(features)
        features, _ = self.cross_modal_attention(features, features, features)
        features = self.post_attn(features)
        
        # Concatenate all features
        combined_features = features.reshape(features.size(0), -1)  # (B, 2056)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return { "logits": logits }

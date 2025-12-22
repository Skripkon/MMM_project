import torch
from torch import nn
from torch.nn import MultiheadAttention

from src.models.backbones.parts import MLP, CNNEncoder, DualPath


class CLIPLSTMModelV2(nn.Module):
    """
    Multi-modal model combining satellite imagery, time series data, and tabular data.
    
    Architecture:
    - CLIP (CNN on satellite image) -> v1 (512,)
    - LSTM 1 (bioclimatic time series) -> v2 (512,)
    - LSTM 2 (landsat time series) -> v3 (512,)
    - MLP (environmental values) -> v4 (512,)
    - Cross-attention: CLIP x LSTM1, LSTM1 x CLIP
    - Concatenate v1, v2, v3, v4 -> (2048,)
    - Final MLP: (2048,) -> (num_classes,)
    """

    def __init__(self, num_classes, hidden_dim=512):
        """
        Args:
            num_classes (int): number of output classes.
            hidden_dim (int): hidden dimension for intermediate representations.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # CLIP: CNN on satellite image (B, 4, 64, 64) -> (512,)
        self.clip_cnn = CNNEncoder(input_channels=4, hidden_dim=256)
        self.clip_fc = nn.Linear(256, hidden_dim)

        # LSTM 1: bioclimatic time series (B, 4, 19, 12) -> (512,)
        # Flatten to (B, 4*19*12) = (B, 912)
        self.lstm1 = DualPath(feature_size=19, time_size=12, num_layers=2)
        self.lstm1_fc = nn.Linear(912, hidden_dim)

        # LSTM 2: landsat time series (B, 6, 4, 21) -> (512,)
        # Flatten to (B, 6*4*21) = (B, 504)
        self.lstm2 = DualPath(feature_size=4, time_size=21, num_layers=2)
        self.lstm2_fc = nn.Linear(504, hidden_dim)

        # MLP: environmental values (B, 5) -> (512,)
        self.mlp_env = MLP(input_dim=5, hidden_dims=[256], output_dim=hidden_dim)

        # Cross-attention modules
        self.cross_attn_clip_lstm1 = MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.cross_attn_lstm1_clip = MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

        # Final MLP: (2048,) -> (num_classes,)
        self.head = MLP(input_dim=hidden_dim * 4, hidden_dims=[1024], output_dim=num_classes)

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

        # CLIP: satellite image -> v1 (B, 512)
        v1 = self.clip_fc(self.clip_cnn(satellite))  # (B, 512)

        # LSTM 1: bioclimatic time series -> v2 (B, 512)
        v2 = self.lstm1_fc(self.lstm1(bioclimatic).reshape(B, -1))  # (B, 512)

        # LSTM 2: landsat time series -> v3 (B, 512)
        v3 = self.lstm2_fc(self.lstm2(landsat).reshape(B, -1))  # (B, 512)

        # MLP: environmental values -> v4 (B, 512)
        v4 = self.mlp_env(table_data)  # (B, 512)

        # Cross-attention: CLIP x LSTM1
        attn_v1, _ = self.cross_attn_clip_lstm1(v1.unsqueeze(1), v2.unsqueeze(1), v2.unsqueeze(1))  # Q=CLIP, K=LSTM1, V=LSTM1 -> (B, 512)
        attn_v1 = attn_v1.squeeze(1)

        # Cross-attention: LSTM1 x CLIP
        attn_v2, _ = self.cross_attn_lstm1_clip(v2.unsqueeze(1), v1.unsqueeze(1), v1.unsqueeze(1))  # Q=LSTM1, K=CLIP, V=CLIP -> (B, 512)
        attn_v2 = attn_v2.squeeze(1)

        # Concatenate: v1, v2, v3, v4 -> (B, 2048)
        concatenated = torch.cat([attn_v1, attn_v2, v3, v4], dim=1)  # (B, 2048)

        # Final MLP: (2048,) -> (n_classes,)
        logits = self.head(concatenated)  # (B, n_classes)

        return {"logits": logits}

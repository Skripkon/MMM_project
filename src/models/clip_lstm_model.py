import torch
from torch import nn
from torch.nn import Sequential


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism.
    """

    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim (int): dimension of query, key, and value.
        """
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, query, key, value):
        """
        Args:
            query (Tensor): query tensor. (B, hidden_dim)
            key (Tensor): key tensor. (B, hidden_dim)
            value (Tensor): value tensor. (B, hidden_dim)
        Returns:
            output (Tensor): attention output. (B, hidden_dim)
        """
        Q = self.query_proj(query)  # (B, hidden_dim)
        K = self.key_proj(key)  # (B, hidden_dim)
        V = self.value_proj(value)  # (B, hidden_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, 1, 1)
        attention_weights = torch.softmax(scores, dim=-1)  # (B, 1, 1)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (B, hidden_dim)
        return output


class CLIPLSTMModel(nn.Module):
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

    def __init__(self, num_classes, hidden_dim=512, lstm_hidden=256, use_for_training_adaptive_k: bool = False):
        """
        Args:
            num_classes (int): number of output classes.
            hidden_dim (int): hidden dimension for intermediate representations.
            lstm_hidden (int): hidden dimension for LSTM.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden

        # CLIP: CNN on satellite image (B, 4, 64, 64) -> (512,)
        self.clip_cnn = Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (128, 16, 16)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (256, 8, 8)
            nn.AdaptiveAvgPool2d((1, 1)),  # (256, 1, 1)
        )
        self.clip_fc = nn.Linear(256, hidden_dim)

        # LSTM 1: bioclimatic time series (B, 4, 19, 12) -> (512,)
        # Flatten to (B, 4*19*12) = (B, 912)
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2, batch_first=True)
        self.lstm1_fc = nn.Linear(lstm_hidden, hidden_dim)

        # LSTM 2: landsat time series (B, 6, 4, 21) -> (512,)
        # Flatten to (B, 6*4*21) = (B, 504)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=2, batch_first=True)
        self.lstm2_fc = nn.Linear(lstm_hidden, hidden_dim)

        # MLP: environmental values (B, 5) -> (512,)
        self.mlp_env = Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

        # Cross-attention modules
        self.cross_attn_clip_lstm1 = CrossAttention(hidden_dim)
        self.cross_attn_lstm1_clip = CrossAttention(hidden_dim)

        # Final MLP: (2048,) -> (num_classes,)
        if not use_for_training_adaptive_k:
            self.final_mlp = Sequential(
                nn.Linear(hidden_dim * 4, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes),
            )
        # Regression task: (2048,) -> (1,)
        else:
            self.final_mlp = Sequential(
                nn.Linear(hidden_dim * 4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1),
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

        # CLIP: satellite image -> v1 (B, 512)
        v1 = self.clip_cnn(satellite)  # (B, 256, 1, 1)
        v1 = v1.view(B, -1)  # (B, 256)
        v1 = self.clip_fc(v1)  # (B, 512)

        # LSTM 1: bioclimatic time series -> v2 (B, 512)
        bio_flat = bioclimatic.view(B, -1, 1)  # (B, 4*19*12, 1) = (B, 912, 1)
        _, (bio_hidden, _) = self.lstm1(bio_flat)  # bio_hidden: (2, B, lstm_hidden)
        v2 = bio_hidden[-1]  # (B, lstm_hidden)
        v2 = self.lstm1_fc(v2)  # (B, 512)

        # LSTM 2: landsat time series -> v3 (B, 512)
        land_flat = landsat.view(B, -1, 1)  # (B, 6*4*21, 1) = (B, 504, 1)
        _, (land_hidden, _) = self.lstm2(land_flat)  # land_hidden: (2, B, lstm_hidden)
        v3 = land_hidden[-1]  # (B, lstm_hidden)
        v3 = self.lstm2_fc(v3)  # (B, 512)

        # MLP: environmental values -> v4 (B, 512)
        v4 = self.mlp_env(table_data)  # (B, 512)

        # Cross-attention: CLIP x LSTM1
        attn_v1 = self.cross_attn_clip_lstm1(v1, v2, v2)  # Q=CLIP, K=LSTM1, V=LSTM1 -> (B, 512)

        # Cross-attention: LSTM1 x CLIP
        attn_v2 = self.cross_attn_lstm1_clip(v2, v1, v1)  # Q=LSTM1, K=CLIP, V=CLIP -> (B, 512)

        # Concatenate: v1, v2, v3, v4 -> (B, 2048)
        concatenated = torch.cat([attn_v1, attn_v2, v3, v4], dim=1)  # (B, 2048)

        # Final MLP: (2048,) -> (n_classes,)
        logits = self.final_mlp(concatenated)  # (B, n_classes)

        return {"logits": logits}

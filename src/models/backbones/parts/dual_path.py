from torch import nn


class DualPath(nn.Module):
    """
    Dual Path module processes input with rnn by two dimensions (time and features).
    """
    def __init__(self, feature_size, time_size, num_layers=1, rnn_type=nn.LSTM, attention_heads=1):
        """
        Args:
            feature_size (int): size of the feature dimension.
            time_size (int): size of the time dimension.
            num_layers (int): number of RNN layers.
            batch_first (bool): whether the input tensor has batch as the first dimension.
        """
        super().__init__()

        self.feature_size = feature_size
        self.time_size = time_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads

        self.rnn_time = rnn_type(input_size=time_size, hidden_size=time_size,
                                num_layers=num_layers, batch_first=True)
        self.rnn_feature = rnn_type(input_size=feature_size, hidden_size=feature_size,
                                   num_layers=num_layers, batch_first=True)
        
        if attention_heads:
            self.multi_domain_attn = nn.MultiheadAttention(embed_dim=feature_size, num_heads=attention_heads, batch_first=True)
        else:
            self.multi_domain_attn = None

    def forward(self, x):
        """
        Forward pass of the Dual Path module.

        Args:
            x (Tensor): input tensor of shape (B, C, F, T).
        Returns:
            output (Tensor): output tensor of shape (B, C, F, T).
        """
        B, C, F, T = x.size()
        # Process along time dimension
        x_time = x.permute(0, 2, 1, 3).contiguous().view(B * F, C, T)  # (B*F, C, T)
        out_time, _ = self.rnn_time(x_time)  # (B*F, C, T)
        out_time = out_time.view(B, F, C, T).permute(0, 2, 1, 3)  # (B, C, F, T)

        # Process along feature dimension
        x_feature = out_time.permute(0, 3, 1, 2).contiguous().view(B * T, C, F)  # (B*T, C, F)
        out_feature, _ = self.rnn_feature(x_feature)  # (B*T, C, F)
        out_feature = out_feature.view(B, T, C, F).permute(0, 2, 3, 1)  # (B, C, F, T)

        # Apply multi-domain attention
        if self.multi_domain_attn is None:
            out_feature = out_feature.view(B, C, F * T).permute(0, 2, 1)  # (B, F*T, C)
            out_feature, _ = self.multi_domain_attn(out_feature, out_feature, out_feature)  # (B, F*T, C)
            out_feature = out_feature.permute(0, 2, 1).view(B, C, F, T)  # (B, C, F, T)

        return out_feature

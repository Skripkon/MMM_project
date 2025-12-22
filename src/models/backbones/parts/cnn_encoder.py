from torch import nn


class CNNEncoder(nn.Module):
    """
    Simple CNN Encoder for image data.
    """

    def __init__(self, input_channels, hidden_dim):
        """
        Args:
            input_channels (int): number of input channels.
            hidden_dim (int): dimension of the hidden representation.
        """
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        """
        Forward pass of the CNN Encoder.

        Args:
            x (Tensor): input tensor of shape (B, C, H, W).
        Returns:
            output (Tensor): output tensor of shape (B, hidden_dim).
        """
        out = self.encoder(x)  # (B, hidden_dim, 1, 1)
        return out.view(out.size(0), -1)  # (B, hidden_dim)

from typing import List, Literal

import torch.nn as nn

from src.models.backbones.parts.resnet_block import ResnetBlock


class ResNet(nn.Module):
    """2x(3x3 Conv + BN + ReLU) with optional downsampling on the skip path."""
    def __init__(
            self,
            in_channels: int,
            hidden_dim: List[int],
            out_channels: int,
            strides: List[int] = [1],
            stem_type: Literal['normal', 'depthpointwise'] = 'normal',
            stem_stride: int = 1,
            use_se: bool = False
        ):
        super().__init__()

        if stem_type == 'normal':
            self.stem = nn.Conv2d(in_channels, hidden_dim[0], kernel_size=3, stride=stem_stride, padding=1)
        elif stem_type == 'depthpointwise':
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim[0], kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(hidden_dim[0], hidden_dim[0], kernel_size=3, stride=stem_stride, padding=1, bias=False)
            )
        self.pre_encoder = nn.Sequential(
            nn.BatchNorm2d(hidden_dim[0]),
            nn.ReLU(),
        )
        self.encoder = nn.ModuleList([])
        for i, (in_c, out_c) in enumerate(zip(hidden_dim, hidden_dim[1:] + [out_channels])):
            self.encoder.append(ResnetBlock(in_c, out_c, stride=strides[i], use_se=use_se))
        self.post_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        """
        Model forward method.

        Args:
            x (Tensor): input data. (B, C, H, W)
        Returns:
            out (Tensor): output features. (B, out_channels)
        """
        x = self.stem(x)
        x = self.pre_encoder(x)
        for block in self.encoder:
            x = block(x)
        out = self.post_encoder(x)
        return out

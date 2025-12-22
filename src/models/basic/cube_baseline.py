import torch.nn as nn

from src.models.backbones.resnet import ResNet
from src.models.backbones.parts import MLP


class CubeBaselineModel(nn.Module):
    """
    ResNet-6 for Landsat cubes:
      - Input: [B, 6, 4, 21]  (bands, quarters, years)
      - Stem: 1x1 conv (channel mixing) + 3x3 conv (spatial-temporal)
      - 3 residual blocks (6 conv layers total)
      - GAP + MLP head -> logits for multi-label BCEWithLogitsLoss
    """
    def __init__(self, num_classes: int, input_type: str, stem_channels: int = 64, mlp_hidden: int = 512, p_drop: float = 0.1):
        super().__init__()

        self.input_type = input_type

        input_shape = {
            'landsat': (6, 4, 21),
            'bioclimatic': (4, 19, 12),
            'sentinel': (4, 64, 64)
        }[input_type]

        # Per-sample normalization over [C,H,W]
        self.norm_input = nn.LayerNorm(input_shape)

        if input_type == 'landsat':
            self.encoder = ResNet(
                in_channels=input_shape[0],
                hidden_dim=[stem_channels, stem_channels, stem_channels],
                out_channels=stem_channels,
                strides=[1, 1, 1],
                stem_type='depthpointwise',
            )
            conv_out_channels = stem_channels
        elif input_type == 'bioclimatic':
            self.encoder = ResNet(
                in_channels=input_shape[0],
                hidden_dim=[stem_channels, stem_channels, stem_channels],
                out_channels=stem_channels,
                strides=[1, 1, 1],
            )
            conv_out_channels = stem_channels
        else:  # sentinel
            self.encoder = ResNet(
                in_channels=input_shape[0],
                hidden_dim=[stem_channels, stem_channels, stem_channels * 2],
                out_channels=stem_channels * 2,
                strides=[1, 2, 1],
                stem_stride=2,
            )
            conv_out_channels = stem_channels * 2

        # Global average pooling and head
        self.head = nn.Sequential(
            nn.LayerNorm(conv_out_channels),
            MLP(
                conv_out_channels,
                [mlp_hidden],
                num_classes,
                dropout=p_drop
            )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

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
        x = {
            'landsat': landsat,
            'bioclimatic': bioclimatic,
            'sentinel': satellite
        }[self.input_type]

        x = self.norm_input(x)  # [B, C, H, W]
        x = self.encoder(x)     # [B, conv_out_channels]
        x = self.head(x)        # [B, num_classes] (logits)
        return x

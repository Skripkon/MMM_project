import torch.nn as nn
import torch.nn.functional as F

from src.models.parts import ResnetBlock


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

        # Two-step stem to (i) mix spectral bands, then (ii) capture local 2D structure
        if input_type == 'landsat':
            self.stem1 = nn.Conv2d(input_shape[0], stem_channels, kernel_size=1, stride=1, padding=0, bias=False)  # channel mixing
            self.stem2 = nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False)
        elif input_type == 'bioclimatic':
            self.stem1 = nn.Identity()
            self.stem2 = nn.Conv2d(input_shape[0], stem_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:  # sentinel
            self.stem1 = nn.Identity()
            self.stem2 = nn.Conv2d(input_shape[0], stem_channels, kernel_size=3, stride=2, padding=1, bias=False)  # 64->32

        self.stem_bn = nn.BatchNorm2d(stem_channels)

        # 3 residual blocks (no downsampling; preserve 4×21)
        if input_type == 'landsat' or input_type == 'bioclimatic':
            self.block1 = ResnetBlock(stem_channels, stem_channels)
            self.block2 = ResnetBlock(stem_channels, stem_channels)
            self.block3 = ResnetBlock(stem_channels, stem_channels)

            conv_out_channels = stem_channels
        else:  # sentinel
            self.block1 = ResnetBlock(stem_channels, stem_channels)
            self.block2 = ResnetBlock(stem_channels, stem_channels*2, stride=2) # 32->16
            self.block3 = ResnetBlock(stem_channels*2, stem_channels*2)

            conv_out_channels = stem_channels*2

        # Global average pooling and head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),                       # [B, C, 1, 1] -> [B, C]
            nn.LayerNorm(conv_out_channels),
            nn.Linear(conv_out_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, num_classes)  # logits
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

        x = self.norm_input(x)
        x = self.stem1(x)                  # [B, C=stem_channels, 4, 21]
        x = self.stem2(x)
        x = self.stem_bn(x)
        x = F.relu(x, inplace=True)

        x = self.block1(x)                 # 3 blocks × 2 conv = 6 conv layers
        x = self.block2(x)
        x = self.block3(x)

        x = self.gap(x)                    # [B, C, 1, 1]
        x = self.head(x)                   # [B, num_classes] (logits)
        return x

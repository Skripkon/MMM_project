from torch.nn import MSELoss


from typing import Any
import torch
from torch import Tensor
from torch.nn import MSELoss, HuberLoss


class MSELossWrapper(MSELoss):

    def __init__(
        self
    ):
        super().__init__()

    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        loss = super().forward(logits.view(-1), target.view(-1))
        return {"loss": loss}


class HuberLossWrapper(HuberLoss):

    def __init__(self):
        super().__init__()

    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        loss = super().forward(logits.view(-1), target.view(-1))
        return {"loss": loss}

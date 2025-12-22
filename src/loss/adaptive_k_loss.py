from torch.nn import MSELoss


from torch import Tensor
from torch.nn import MSELoss, HuberLoss


class MSELossWrapper(MSELoss):

    def __init__(
        self
    ):
        super().__init__()

    def forward(self, k_pred: Tensor, k: Tensor, **batch) -> dict[str, Tensor]:
        loss = super().forward(k_pred.view(-1), k.view(-1))
        return {"loss": loss}


class HuberLossWrapper(HuberLoss):

    def __init__(self):
        super().__init__()

    def forward(self, k_pred: Tensor, k: Tensor, **batch) -> dict[str, Tensor]:
        loss = super().forward(k_pred.view(-1), k.view(-1))
        return {"loss": loss}

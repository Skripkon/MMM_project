from torch import nn
from torch.nn import Sequential


class MLPBaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, num_classes, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            num_classes (int): number of output classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=num_classes),
        )

    def forward(self, satellite, bioclimatic, landsat, table_data, **batch):
        """
        Model forward method.

        Args:
            satellite (Tensor): satellite data. (B, Cs, Hs, Ws)
            bioclimatic (Tensor): bioclimatic data. (B, Cb, Hb, Wb)
            landsat (Tensor): landsat data. (B, Cl, Hl, Wl)
            table_data (Tensor): tabular data. (B, Ft)
            **batch: other batch data.
        Returns:
            output (dict): output dict containing log_probs.
        Note:
            Cs, Hs, Ws = 4, 64, 64
            Cb, Hb, Wb = 4, 19, 12
            Cl, Hl, Wl = 6, 4, 21
            Ft = 5
        """

        x = table_data.view(table_data.size(0), -1)  # flatten
        x = self.net(x)
        return {"logits": x}

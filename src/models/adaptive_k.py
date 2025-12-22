from torch import nn

from src.models.backbones.parts import MLP


class AdaptiveK(nn.Module):
    """Multi-modal fusion model with cross-modal attention"""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        
        self.backbone = backbone
        
        # Final classifier
        self.head = MLP(
            input_dim=backbone.head.input_dim,
            hidden_dims=[1024],
            output_dim=1,
            dropout=0.3
        )

    def forward(self, **batch):
        """
        Model forward method.

        Args:
            **batch: other batch data.
        Returns:
            output (dict): output dict containing k_pred.
        """
        backbone_head = self.backbone.head  # Save original head
        self.backbone.head = nn.Identity()
        hidden = self.backbone(**batch)["logits"]  # (B, D)
        self.backbone.head = backbone_head
        
        # Classification
        k = self.head(hidden).flatten()  # (B,)
        
        return { "k_pred": k }
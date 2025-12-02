from typing import Any
import torch
from torch import Tensor
from torch.nn import Module, BCEWithLogitsLoss, CrossEntropyLoss, SoftMarginLoss


class BCELossWrapper(BCEWithLogitsLoss):
    """
    Binary Cross Entropy Loss for multi-label classification.
    Treats each label as an independent binary classification task.
    """
    def __init__(
        self,
        weight: Tensor | None = None,
        size_average: Any | None = None,
        reduce: Any | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None
    ):
        super().__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight)

    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            logits: Predicted logits of shape (batch_size, num_labels)
            target: Ground truth binary labels of shape (batch_size, num_labels)
            
        Returns:
            Dictionary with 'loss' key containing the loss value
        """
        loss = super().forward(logits.view(-1), target.view(-1))
        return {"loss": loss}


class AsymmetricLossWrapper(Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Based on "Asymmetric Loss For Multi-Label Classification" (https://arxiv.org/abs/2009.14119)
    
    Asymmetric Loss addresses the class imbalance problem by:
    - Focusing more on false positives (easier samples) with exponential dampening
    - Focusing more on hard negatives with power-law focusing
    - Using different loss weights for positive and negative samples
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        *args,
        **kwargs
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples (hard negatives)
            gamma_pos: Focusing parameter for positive samples
            clip: Gradient clipping threshold to prevent positive gradients
            eps: Small value for numerical stability
        """
        super().__init__(*args, **kwargs)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            logits: Predicted logits of shape (batch_size, num_labels)
            target: Ground truth binary labels of shape (batch_size, num_labels)
            
        Returns:
            Dictionary with 'loss' key containing the loss value
        """
        p = torch.sigmoid(logits)
        p = torch.clamp(p, min=self.eps, max=1 - self.eps)
        
        p_pos = p[target == 1]
        loss_pos = -torch.pow(1 - p_pos, self.gamma_pos) * torch.log(p_pos + self.eps)
        
        p_neg = p[target == 0]
        loss_neg = -torch.pow(p_neg, self.gamma_neg) * torch.log(1 - p_neg + self.eps)
        
        if self.clip > 0:
            loss_pos = torch.clamp(loss_pos, max=self.clip)
        
        loss = (loss_pos.sum() + loss_neg.sum()) / (target.shape[0] * target.shape[1])
        
        return {"loss": loss}


class FocalLossWrapper(Module):
    """
    Focal Loss for multi-label classification.
    
    Addresses class imbalance by down-weighting easy negative examples and 
    focusing training on hard negatives and positives.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        *args,
        **kwargs
    ):
        """
        Args:
            alpha: Weighting factor in range (0,1) to balance positive vs negative examples
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples
            reduction: 'mean' or 'sum'
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            logits: Predicted logits of shape (batch_size, num_labels)
            target: Ground truth binary labels of shape (batch_size, num_labels)
            
        Returns:
            Dictionary with 'loss' key containing the loss value
        """
        p = torch.sigmoid(logits)
        target = target.float()
        
        p_t = torch.where(target == 1, p, 1 - p)
        focal_weight = torch.pow(1 - p_t, self.gamma)
        
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target.float(), reduction='none'
        )
        
        loss = self.alpha * focal_weight * bce
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return {"loss": loss}


class SoftMarginLossWrapper(SoftMarginLoss):
    """
    Soft Margin Loss for multi-label classification.
    Uses a smooth approximation to the classification error.
    """
    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            logits: Predicted logits of shape (batch_size, num_labels)
            target: Ground truth labels {-1, 1} or {0, 1} of shape (batch_size, num_labels)
            
        Returns:
            Dictionary with 'loss' key containing the loss value
        """
        # Convert {0, 1} targets to {-1, 1} if needed
        target_transformed = torch.where(target > 0.5, torch.ones_like(target), -torch.ones_like(target))
        
        loss = super().forward(logits, target_transformed)
        
        return {"loss": loss}


class MultiLabelSmoothingLossWrapper(BCEWithLogitsLoss):
    """
    Binary Cross Entropy Loss with label smoothing for multi-label classification.
    Smoothing helps prevent overconfident predictions.
    """
    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Tensor | None = None,
        size_average: Any | None = None,
        reduce: Any | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
        *args,
        **kwargs
    ):
        """
        Args:
            smoothing: Label smoothing parameter in range [0, 1]
            reduction: 'mean' or 'sum'
        """
        super().__init__(weight=weight, size_average=size_average, reduce=reduce, reduction=reduction, pos_weight=pos_weight, *args, **kwargs)
        self.smoothing = smoothing

    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            logits: Predicted logits of shape (batch_size, num_labels)
            target: Ground truth binary labels of shape (batch_size, num_labels)
            
        Returns:
            Dictionary with 'loss' key containing the loss value
        """
        smoothed_target = target.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        
        loss = super().forward(logits, smoothed_target)
        
        return {"loss": loss}


class MarginLossWrapper(Module):
    """
    Margin-based Loss for multi-label classification.
    Learns an explicit margin between positive and negative examples.
    """
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = "mean",
        *args,
        **kwargs
    ):
        """
        Args:
            margin: Margin value for separation
            reduction: 'mean' or 'sum'
        """
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor, **batch) -> dict[str, Tensor]:
        """
        Args:
            logits: Predicted logits of shape (batch_size, num_labels)
            target: Ground truth binary labels of shape (batch_size, num_labels)
            
        Returns:
            Dictionary with 'loss' key containing the loss value
        """
        loss_pos = torch.clamp(self.margin - logits[target == 1], min=0)
        loss_neg = torch.clamp(logits[target == 0] + self.margin, min=0)
        
        loss = (loss_pos.sum() + loss_neg.sum()) / (target.shape[0] * target.shape[1])
        
        if self.reduction == "sum":
            loss = loss * (target.shape[0] * target.shape[1])
        
        return {"loss": loss}

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import (
    calc_multilabel_accuracy,
    calc_multilabel_precision,
    calc_multilabel_recall,
    calc_multilabel_f1,
    calc_multilabel_hamming_loss,
    calc_multilabel_jaccard_similarity,
)


class Accuracy(BaseMetric):
    """
    Exact match accuracy for multi-label classification.
    Computes the percentage of samples where all labels are predicted exactly correctly.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: Tensor, target: Tensor, threshold: float = 0.5, **batch):
        return calc_multilabel_accuracy(logits, target, threshold)


class Precision(BaseMetric):
    """
    Precision for multi-label classification.
    Macro-averaged precision across all labels.
    """
    def __init__(self, average: str = "macro", *args, **kwargs):
        """
        Args:
            average (str): Type of averaging - "macro", "micro", or "weighted".
        """
        super().__init__(*args, **kwargs)
        self.average = average

    def __call__(self, logits: Tensor, target: Tensor, threshold: float = 0.5, **batch):
        return calc_multilabel_precision(logits, target, threshold, self.average)


class Recall(BaseMetric):
    """
    Recall for multi-label classification.
    Macro-averaged recall across all labels.
    """
    def __init__(self, average: str = "macro", *args, **kwargs):
        """
        Args:
            average (str): Type of averaging - "macro", "micro", or "weighted".
        """
        super().__init__(*args, **kwargs)
        self.average = average

    def __call__(self, logits: Tensor, target: Tensor, threshold: float = 0.5, **batch):
        return calc_multilabel_recall(logits, target, threshold, self.average)


class F1Score(BaseMetric):
    """
    F1 Score for multi-label classification.
    Macro-averaged F1 across all labels.
    """
    def __init__(self, average: str = "macro", *args, **kwargs):
        """
        Args:
            average (str): Type of averaging - "macro", "micro", or "weighted".
        """
        super().__init__(*args, **kwargs)
        self.average = average

    def __call__(self, logits: Tensor, target: Tensor, threshold: float = 0.5, **batch):
        return calc_multilabel_f1(logits, target, threshold, self.average)


class HammingLoss(BaseMetric):
    """
    Hamming Loss for multi-label classification.
    Fraction of incorrectly predicted labels.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: Tensor, target: Tensor, threshold: float = 0.5, **batch):
        return calc_multilabel_hamming_loss(logits, target, threshold)


class JaccardSimilarity(BaseMetric):
    """
    Jaccard Similarity (Intersection over Union) for multi-label classification.
    Macro-averaged IoU across all samples.
    """
    def __init__(self, average: str = "macro", *args, **kwargs):
        """
        Args:
            average (str): Type of averaging - "macro" or "micro".
        """
        super().__init__(*args, **kwargs)
        self.average = average

    def __call__(self, logits: Tensor, target: Tensor, threshold: float = 0.5, **batch):
        return calc_multilabel_jaccard_similarity(logits, target, threshold, self.average)

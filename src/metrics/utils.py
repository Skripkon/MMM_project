import torch
from torch import Tensor


def _convert_logits_to_binary(logits: Tensor, threshold: float) -> Tensor:
    """
    Convert logits to binary predictions based on a threshold.
    
    Args:
        logits: Predicted probabilities or logits of shape (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Binary predictions of shape (batch_size, num_labels)
    """
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()


def calc_multilabel_accuracy(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """
    Calculate exact match accuracy for multi-label classification.
    Returns 1 only if all labels are predicted correctly for a sample.
    
    Args:
        logits: Predicted probabilities or logits of shape (batch_size, num_labels)
        targets: Ground truth binary labels of shape (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Exact match accuracy as a float
    """
    # Convert logits to probabilities if needed
    preds_binary = _convert_logits_to_binary(logits, threshold)
    
    # Exact match: all labels must match
    exact_match = (preds_binary == targets).all(dim=1).float()
    return exact_match.mean().item()


def calc_multilabel_precision(
    logits: Tensor, targets: Tensor, threshold: float = 0.5, average: str = "macro"
) -> float:
    """
    Calculate precision for multi-label classification.
    
    Args:
        logits: Predicted probabilities or logits of shape (batch_size, num_labels)
        targets: Ground truth binary labels of shape (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        average: "macro", "micro", or "weighted" averaging
        
    Returns:
        Precision score as a float
    """
    preds_binary = _convert_logits_to_binary(logits, threshold)
    
    if average == "micro":
        # Global precision: TP / (TP + FP)
        tp = (preds_binary * targets).sum()
        fp = (preds_binary * (1 - targets)).sum()
        precision = tp / (tp + fp + 1e-8)
        return precision.item()
    
    elif average == "macro":
        # Per-label precision, then average
        tp = (preds_binary * targets).sum(dim=0)
        fp = (preds_binary * (1 - targets)).sum(dim=0)
        precisions = tp / (tp + fp + 1e-8)
        return precisions.mean().item()
    
    elif average == "weighted":
        # Weighted by number of positives per label
        tp = (preds_binary * targets).sum(dim=0)
        fp = (preds_binary * (1 - targets)).sum(dim=0)
        precisions = tp / (tp + fp + 1e-8)
        weights = targets.sum(dim=0) / targets.sum()
        return (precisions * weights).sum().item()
    
    else:
        raise ValueError(f"Unknown average method: {average}")


def calc_multilabel_recall(
    logits: Tensor, targets: Tensor, threshold: float = 0.5, average: str = "macro"
) -> float:
    """
    Calculate recall for multi-label classification.
    
    Args:
        logits: Predicted probabilities or logits of shape (batch_size, num_labels)
        targets: Ground truth binary labels of shape (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        average: "macro", "micro", or "weighted" averaging
        
    Returns:
        Recall score as a float
    """
    preds_binary = _convert_logits_to_binary(logits, threshold)
    
    if average == "micro":
        # Global recall: TP / (TP + FN)
        tp = (preds_binary * targets).sum()
        fn = ((1 - preds_binary) * targets).sum()
        recall = tp / (tp + fn)
        return recall.item()
    
    elif average == "macro":
        # Per-label recall, then average
        tp = (preds_binary * targets).sum(dim=-1)
        fn = ((1 - preds_binary) * targets).sum(dim=-1)
        recalls = tp / (tp + fn)
        return recalls.mean().item()
    
    elif average == "weighted":
        # Weighted by number of positives per label
        tp = (preds_binary * targets).sum(dim=-1)
        fn = ((1 - preds_binary) * targets).sum(dim=-1)
        recalls = tp / (tp + fn)
        weights = targets.sum(dim=0) / targets.sum()
        return (recalls * weights).sum().item()
    
    else:
        raise ValueError(f"Unknown average method: {average}")


def calc_multilabel_f1(
    logits: Tensor, targets: Tensor, threshold: float = 0.5, average: str = "macro"
) -> float:
    """
    Calculate F1 score for multi-label classification.
    
    Args:
        logits: Predicted probabilities or logits of shape (batch_size, num_labels)
        targets: Ground truth binary labels of shape (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        average: "macro", "micro", or "weighted" averaging
        
    Returns:
        F1 score as a float
    """
    preds_binary = _convert_logits_to_binary(logits, threshold)
    
    if average == "micro":
        # Global F1
        tp = (preds_binary * targets).sum()
        fp = (preds_binary * (1 - targets)).sum()
        fn = ((1 - preds_binary) * targets).sum()
        f1s = tp / (tp + (fp + fn) / 2)
        return f1s.item()
    
    elif average == "macro":
        # Per-label F1, then average
        tp = (preds_binary * targets).sum(dim=-1)
        fp = (preds_binary * (1 - targets)).sum(dim=-1)
        fn = ((1 - preds_binary) * targets).sum(dim=-1)
        f1s = tp / (tp + (fp + fn) / 2)
        return f1s.mean().item()
    
    elif average == "weighted":
        # Weighted by number of positives per label
        tp = (preds_binary * targets).sum(dim=-1)
        fp = (preds_binary * (1 - targets)).sum(dim=-1)
        fn = ((1 - preds_binary) * targets).sum(dim=-1)
        f1s = tp / (tp + (fp + fn) / 2)
        weights = targets.sum(dim=0) / targets.sum()
        return (f1s * weights).sum().item()
    
    else:
        raise ValueError(f"Unknown average method: {average}")


def calc_multilabel_hamming_loss(logits: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """
    Calculate Hamming Loss for multi-label classification.
    Represents the fraction of incorrectly predicted labels.
    Lower is better.
    
    Args:
        logits: Predicted probabilities or logits of shape (batch_size, num_labels)
        targets: Ground truth binary labels of shape (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Hamming Loss as a float (between 0 and 1)
    """
    preds_binary = _convert_logits_to_binary(logits, threshold)
    
    # Fraction of incorrectly labeled instances
    hamming_loss = (preds_binary != targets).float().mean()
    return hamming_loss.item()


def calc_multilabel_jaccard_similarity(
    logits: Tensor, targets: Tensor, threshold: float = 0.5, average: str = "macro"
) -> float:
    """
    Calculate Jaccard Similarity (IoU) for multi-label classification.
    For each sample: |intersection| / |union| of predicted and target labels.
    
    Args:
        logits: Predicted probabilities or logits of shape (batch_size, num_labels)
        targets: Ground truth binary labels of shape (batch_size, num_labels)
        threshold: Threshold for converting probabilities to binary predictions
        average: "macro" or "micro" averaging
        
    Returns:
        Jaccard Similarity as a float (between 0 and 1)
    """
    preds_binary = _convert_logits_to_binary(logits, threshold)
    
    if average == "macro":
        # Per-sample Jaccard, then average
        intersection = (preds_binary * targets).sum(dim=1)
        union = ((preds_binary + targets) > 0).float().sum(dim=1)
        jaccard = intersection / (union + 1e-8)
        return jaccard.mean().item()
    
    elif average == "micro":
        # Global Jaccard
        intersection = (preds_binary * targets).sum()
        union = ((preds_binary + targets) > 0).float().sum()
        jaccard = intersection / (union + 1e-8)
        return jaccard.item()
    
    else:
        raise ValueError(f"Unknown average method: {average}")

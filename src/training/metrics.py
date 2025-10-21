from __future__ import annotations
import torch


def masked_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute token-level accuracy on positions where labels != ignore_index.
    Args:
      logits: (B, T, V)
      labels: (B, T)
    Returns:
      accuracy in [0,1]; returns 0.0 if no valid positions
    """
    with torch.no_grad():
        B, T, V = logits.shape
        preds = logits.argmax(dim=-1)
        mask = labels != ignore_index
        denom = mask.sum().item()
        if denom == 0:
            return 0.0
        num = (preds[mask] == labels[mask]).sum().item()
        return num / denom

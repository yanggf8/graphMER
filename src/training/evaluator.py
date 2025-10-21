"""Evaluation metrics for MNM/MLM tasks"""
import torch
import numpy as np
from typing import Dict, List, Tuple

def compute_mrr(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """Compute Mean Reciprocal Rank"""
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return 0.0
    
    # Get predictions and filter valid positions
    probs = torch.softmax(logits, dim=-1)
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]
    valid_probs = probs[valid_mask]
    
    # Rank predictions
    _, ranked_indices = torch.sort(valid_probs, dim=-1, descending=True)
    
    reciprocal_ranks = []
    for i, target in enumerate(valid_targets):
        rank_pos = (ranked_indices[i] == target).nonzero(as_tuple=True)[0]
        if len(rank_pos) > 0:
            reciprocal_ranks.append(1.0 / (rank_pos[0].item() + 1))
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_hits_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 10, ignore_index: int = -100) -> float:
    """Compute Hits@K"""
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return 0.0
    
    probs = torch.softmax(logits, dim=-1)
    valid_probs = probs[valid_mask]
    valid_targets = targets[valid_mask]
    
    # Get top-k predictions
    _, top_k_indices = torch.topk(valid_probs, k=min(k, valid_probs.size(-1)), dim=-1)
    
    hits = 0
    for i, target in enumerate(valid_targets):
        if target in top_k_indices[i]:
            hits += 1
    
    return hits / len(valid_targets)

def evaluate_model(model, mlm_head, mnm_head, dataloader, device) -> Dict[str, float]:
    """Run evaluation and return metrics"""
    model.eval()
    mlm_head.eval()
    mnm_head.eval()
    
    total_mlm_mrr = 0.0
    total_mnm_mrr = 0.0
    total_mlm_hits = 0.0
    total_mnm_hits = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attn, mlm_labels, mnm_labels, rel_ids = [x.to(device) for x in batch]
            hidden = model(input_ids.unsqueeze(0), attn.unsqueeze(0), rel_ids.unsqueeze(0))
            
            mlm_logits = mlm_head(hidden)
            mnm_logits = mnm_head(hidden)
            
            # Compute metrics for this batch
            total_mlm_mrr += compute_mrr(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            total_mlm_hits += compute_hits_at_k(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1), k=10)
            total_mnm_mrr += compute_mrr(mnm_logits.view(-1, mnm_logits.size(-1)), mnm_labels.view(-1))
            total_mnm_hits += compute_hits_at_k(mnm_logits.view(-1, mnm_logits.size(-1)), mnm_labels.view(-1), k=10)
            count += 1
    
    return {
        "mlm_mrr": total_mlm_mrr / max(count, 1),
        "mlm_hits_at_10": total_mlm_hits / max(count, 1),
        "mnm_mrr": total_mnm_mrr / max(count, 1),
        "mnm_hits_at_10": total_mnm_hits / max(count, 1),
    }

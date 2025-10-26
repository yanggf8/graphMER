"""
Ontology constraint regularizers for GraphMER-SE training.
Implements antisymmetry, acyclicity, and contrastive losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ConstraintLoss(nn.Module):
    """Ontology constraint regularizers.
    Uses only embeddings and relation masks to avoid coupling to vocab logits.
    """
    
    def __init__(self, 
                 antisymmetry_weight: float = 0.2,
                 acyclicity_weight: float = 0.2,
                 contrastive_weight: float = 0.07,
                 antisymmetric_relations: Optional[List[int]] = None,
                 acyclic_relations: Optional[List[int]] = None):
        super().__init__()
        self.antisymmetry_weight = antisymmetry_weight
        self.acyclicity_weight = acyclicity_weight
        self.contrastive_weight = contrastive_weight
        
        # Default SE antisymmetric relations (inherits_from, implements)
        self.antisymmetric_relations = antisymmetric_relations or [1, 2]  # relation IDs
        self.acyclic_relations = acyclic_relations or [1, 2, 3]  # inherits_from, implements, contains
    
    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)  # B,T,1
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom
    
    def antisymmetry_loss(self, embeddings: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Encourage distinct relation-specific representations by penalizing
        high similarity between different relations' pooled embeddings within a batch.
        This serves as a proxy regularizer in absence of explicit (A,R,B)/(B,R,A) pairs.
        """
        if embeddings is None or rel_ids is None:
            return torch.tensor(0.0, device=embeddings.device if embeddings is not None else 'cpu')
        B, T, C = embeddings.shape
        uniq = torch.unique(rel_ids)
        rel_means = []
        rel_list = []
        for r in uniq.tolist():
            if r == 0:
                continue
            mask = (rel_ids == r)  # B,T
            if not mask.any():
                continue
            pooled = self._masked_mean(embeddings, mask)  # B,C
            rel_means.append(pooled.mean(dim=0, keepdim=True))  # 1,C
            rel_list.append(r)
        if len(rel_means) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        means = torch.cat(rel_means, dim=0)  # R,C
        means = torch.nn.functional.normalize(means, dim=-1)
        sims = means @ means.t()  # R,R
        i, j = torch.triu_indices(sims.size(0), sims.size(1), offset=1)
        penalty = sims[i, j].mean()  # encourage low similarity across relations
        return penalty * self.antisymmetry_weight
    
    def acyclicity_loss(self, embeddings: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Within each relation chain, discourage near-identity between early and late segments
        (a crude proxy for cycle avoidance encouraging directional change along the chain).
        """
        if embeddings is None or rel_ids is None:
            return torch.tensor(0.0, device=embeddings.device if embeddings is not None else 'cpu')
        B, T, C = embeddings.shape
        total = torch.tensor(0.0, device=embeddings.device)
        count = 0
        for r in torch.unique(rel_ids).tolist():
            if r == 0:
                continue
            mask = (rel_ids == r)  # B,T
            if not mask.any():
                continue
            # For each batch element, split tokens of this relation into two halves by position
            for b in range(B):
                idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
                if idx.numel() < 4:
                    continue
                half = idx.numel() // 2
                first_idx = idx[:half]
                second_idx = idx[half:]
                first = embeddings[b, first_idx].mean(dim=0)
                second = embeddings[b, second_idx].mean(dim=0)
                first = torch.nn.functional.normalize(first, dim=0)
                second = torch.nn.functional.normalize(second, dim=0)
                sim = torch.dot(first, second)
                total = total + sim
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=embeddings.device)
        return (total / count) * self.acyclicity_weight
    
    def contrastive_loss(self, embeddings: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Contrastive loss: bring tokens of same relation closer and push different relations apart.
        Uses supervised contrastive variant with in-batch positives.
        """
        if embeddings is None or rel_ids is None:
            return torch.tensor(0.0, device=embeddings.device if embeddings is not None else 'cpu')
        B, T, C = embeddings.shape
        x = embeddings.view(B * T, C)
        y = rel_ids.view(B * T)
        mask_valid = y > 0
        x = x[mask_valid]
        y = y[mask_valid]
        if x.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        x = torch.nn.functional.normalize(x, dim=-1)
        sim = x @ x.t()  # N,N
        N = sim.size(0)
        # build supervised contrastive loss
        labels = y.unsqueeze(0) == y.unsqueeze(1)  # N,N
        # Exclude self
        eye = torch.eye(N, device=sim.device, dtype=torch.bool)
        labels = labels & (~eye)
        # Temperature
        tau = max(1e-6, float(self.contrastive_weight))
        logits = sim / tau
        # For each anchor i, positives P(i)
        exp_logits = torch.exp(logits)
        # denominator: sum over j != i
        denom = (exp_logits * (~eye)).sum(dim=1, keepdim=True).clamp_min(1e-6)
        # numerator: sum over positives
        num = (exp_logits * labels.float()).sum(dim=1)
        loss_vec = -torch.log((num.clamp_min(1e-6)) / denom.squeeze(1))
        return loss_vec.mean()
    
    def forward(self, 
                embeddings: torch.Tensor, 
                rel_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all constraint losses from hidden embeddings and relation ids.
        Args:
            embeddings: [B, T, C] hidden states
            rel_ids:    [B, T] relation ids per token (0 = none)
        """
        losses = {}
        losses['antisymmetry'] = self.antisymmetry_loss(embeddings, rel_ids)
        losses['acyclicity'] = self.acyclicity_loss(embeddings, rel_ids)
        losses['contrastive'] = self.contrastive_loss(embeddings, rel_ids)
        losses['total'] = losses['antisymmetry'] + losses['acyclicity'] + losses['contrastive']
        return losses

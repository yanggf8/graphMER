"""
Multi-hop Reasoning for GraphMER - Extends attention to capture multi-hop graph patterns.
Implements path-aware attention that can reason over graph paths of length > 1.
"""
import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple, Optional


class MultiHopRelationAttention(nn.Module):
    """Multi-hop relation-aware attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, num_relations: int, max_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_hops = max_hops
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-hop relation biases
        self.hop_biases = nn.ModuleList([
            nn.Embedding(num_relations, 1) for _ in range(max_hops)
        ])
        
        # Path composition weights
        self.path_weights = nn.Parameter(torch.ones(max_hops) / max_hops)
        
        # Initialize biases
        for hop_bias in self.hop_biases:
            nn.init.zeros_(hop_bias.weight)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, rel_ids: torch.Tensor = None):
        """
        Multi-hop attention forward pass.
        
        Args:
            x: Input embeddings [B, T, D]
            attn_mask: Attention mask [B, T]
            rel_ids: Relation IDs [B, T]
            
        Returns:
            Attention output [B, T, D]
        """
        B, T, C = x.shape
        
        # Standard attention computation
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B,H,T,D
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # B,H,T,T
        
        # Add multi-hop relation biases
        if rel_ids is not None:
            multihop_bias = self._compute_multihop_bias(rel_ids)  # B,T,T
            attn_logits = attn_logits + multihop_bias.unsqueeze(1)  # Broadcast over heads
        
        # Apply attention mask
        if attn_mask is not None:
            mask_value = -1e9
            # Convert mask to attention mask format
            attn_mask_expanded = attn_mask.unsqueeze(1).unsqueeze(1)  # B,1,1,T
            attn_mask_expanded = attn_mask_expanded.expand(B, 1, T, T)
            attn_logits = attn_logits.masked_fill(attn_mask_expanded == 0, mask_value)
        
        # Compute attention weights and output
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # B,H,T,D
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(attn_output)
    
    def _compute_multihop_bias(self, rel_ids: torch.Tensor) -> torch.Tensor:
        """Compute multi-hop relation bias matrix."""
        B, T = rel_ids.shape
        device = rel_ids.device
        
        # Initialize bias matrix
        bias_matrix = torch.zeros(B, T, T, device=device)
        
        # Compute biases for each hop level
        for hop in range(self.max_hops):
            hop_bias = self._compute_hop_bias(rel_ids, hop + 1)
            weight = torch.softmax(self.path_weights, dim=0)[hop]
            bias_matrix = bias_matrix + weight * hop_bias
        
        return bias_matrix
    
    def _compute_hop_bias(self, rel_ids: torch.Tensor, hop_length: int) -> torch.Tensor:
        """Compute bias for specific hop length."""
        B, T = rel_ids.shape
        device = rel_ids.device
        
        bias_matrix = torch.zeros(B, T, T, device=device)
        
        for b in range(B):
            for i in range(T):
                for j in range(T):
                    if i == j:
                        continue
                    
                    # Check if there's a path of length hop_length from i to j
                    path_bias = self._find_path_bias(rel_ids[b], i, j, hop_length)
                    bias_matrix[b, i, j] = path_bias
        
        return bias_matrix
    
    def _find_path_bias(self, rel_ids: torch.Tensor, start: int, end: int, hop_length: int) -> float:
        """Find bias for path between start and end positions."""
        if hop_length == 1:
            # Direct connection
            rel_id = rel_ids[start].item()
            if rel_id > 0 and rel_ids[end].item() == rel_id:
                return self.hop_biases[0](torch.tensor(rel_id, device=rel_ids.device)).item()
            return 0.0
        
        elif hop_length == 2:
            # Two-hop connection: find intermediate nodes
            for mid in range(len(rel_ids)):
                if mid == start or mid == end:
                    continue
                
                # Check path: start -> mid -> end
                rel1 = rel_ids[start].item()
                rel2 = rel_ids[mid].item()
                
                if (rel1 > 0 and rel_ids[mid].item() == rel1 and 
                    rel2 > 0 and rel_ids[end].item() == rel2):
                    bias1 = self.hop_biases[0](torch.tensor(rel1, device=rel_ids.device)).item()
                    bias2 = self.hop_biases[1](torch.tensor(rel2, device=rel_ids.device)).item()
                    return (bias1 + bias2) * 0.5  # Average path bias
            
            return 0.0
        
        else:
            # Higher-order hops (simplified)
            return 0.0


class MultiHopEncoderLayer(nn.Module):
    """Encoder layer with multi-hop attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_relations: int, 
                 max_hops: int = 3, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHopRelationAttention(d_model, n_heads, num_relations, max_hops, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, rel_ids: torch.Tensor = None):
        # Multi-hop self-attention
        attn_out = self.self_attn(x, attn_mask=attn_mask, rel_ids=rel_ids)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


def test_multihop_attention():
    """Test multi-hop attention implementation."""
    print("Testing Multi-hop Attention...")
    
    # Test parameters
    B, T, D = 2, 12, 256
    n_heads = 4
    num_relations = 5
    max_hops = 3
    
    # Create test data with multi-hop structure
    x = torch.randn(B, T, D)
    rel_ids = torch.tensor([
        [0, 0, 1, 1, 2, 2, 1, 1, 3, 3, 0, 0],  # Chain: 1->2->1->3
        [0, 1, 1, 2, 2, 2, 3, 3, 1, 1, 0, 0]   # Different pattern
    ])
    attn_mask = torch.ones(B, T)
    
    # Test multi-hop attention
    multihop_attn = MultiHopRelationAttention(D, n_heads, num_relations, max_hops)
    output = multihop_attn(x, attn_mask, rel_ids)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test encoder layer
    encoder_layer = MultiHopEncoderLayer(D, n_heads, D*4, num_relations, max_hops)
    layer_output = encoder_layer(x, attn_mask, rel_ids)
    
    print(f"✅ Encoder layer output shape: {layer_output.shape}")
    
    assert output.shape == x.shape, "Output shape should match input"
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    
    print("✅ Multi-hop attention test passed!")


if __name__ == "__main__":
    test_multihop_attention()

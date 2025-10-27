from __future__ import annotations
import torch
import torch.nn as nn
import math
from .graph_positional import GraphPositionalEncoding
from .multihop_attention import MultiHopEncoderLayer


class TinyRelSelfAttention(nn.Module):
    """Self-attention with relation-aware bias added to attention logits.
    The bias is built from per-token rel_ids:
      - For tokens sharing the same relation id r (>0), add bias_leaf[r]
      - For cross attention between relation tokens and code tokens (rel_id==0), add bias_cross[r]
    """
    def __init__(self, d_model: int, n_heads: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # relation biases (head-agnostic for simplicity)
        self.bias_leaf = nn.Embedding(num_relations, 1)
        self.bias_cross = nn.Embedding(num_relations, 1)
        nn.init.zeros_(self.bias_leaf.weight)
        nn.init.zeros_(self.bias_cross.weight)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, rel_ids: torch.Tensor | None = None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # B,H,T,D
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # B,H,T,T

        # Build relation bias
        if rel_ids is not None:
            # rel_ids: B,T -> B,1,T and B,T,1 for broadcasting
            rel = rel_ids
            zero = (rel == 0)
            bias = torch.zeros((B, 1, T, T), device=x.device, dtype=attn_logits.dtype)
            # Iterate over present relation ids per batch (small)
            # For simplicity we loop over unique ids in the whole batch
            unique = torch.unique(rel)
            for r in unique.tolist():
                if r == 0:
                    continue
                r_mask_i = (rel == r).unsqueeze(-1)  # B,T,1
                r_mask_j = (rel == r).unsqueeze(-2)  # B,1,T
                # leaf cohesion: same relation
                b_leaf = self.bias_leaf(torch.tensor([r], device=x.device)).view(1, 1, 1, 1)
                bias = bias + (r_mask_i & r_mask_j).unsqueeze(1).to(bias.dtype) * b_leaf
                # cross: relation to code or code to relation
                code_j = zero.unsqueeze(-2)  # B,1,T
                code_i = zero.unsqueeze(-1)  # B,T,1
                b_cross = self.bias_cross(torch.tensor([r], device=x.device)).view(1, 1, 1, 1)
                cross_ij = (r_mask_i & code_j).unsqueeze(1)
                cross_ji = (code_i & r_mask_j).unsqueeze(1)
                bias = bias + cross_ij.to(bias.dtype) * b_cross + cross_ji.to(bias.dtype) * b_cross
            attn_logits = attn_logits + bias

        if attn_mask is not None:
            # attn_mask: B,T where 1 is keep, 0 is pad
            key_padding_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)  # B,1,1,T
            attn_logits = attn_logits.masked_fill(key_padding_mask, float('-inf'))

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # B,H,T,D
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class TinyEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_relations: int, dropout: float = 0.1, use_rel_bias: bool = True):
        super().__init__()
        if use_rel_bias:
            self.self_attn = TinyRelSelfAttention(d_model, n_heads, num_relations, dropout)
        else:
            # fallback to standard MultiheadAttention-like block by reusing TinyRelSelfAttention with zeros
            self.self_attn = TinyRelSelfAttention(d_model, n_heads, num_relations, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, rel_ids: torch.Tensor | None = None):
        attn_out = self.self_attn(x, attn_mask=attn_mask, rel_ids=rel_ids)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32000, d_model: int = 256, n_heads: int = 4, n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1, num_relations: int = 64, use_rel_attention_bias: bool = True, use_multihop: bool = False, max_hops: int = 3):
        super().__init__()
        self.use_rel_attention_bias = use_rel_attention_bias
        self.use_multihop = use_multihop
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        
        # Replace standard positional encoding with graph-aware version
        self.graph_pos_encoding = GraphPositionalEncoding(d_model, max_seq_len=4096)
        
        self.rel_emb = nn.Embedding(num_relations, d_model)  # embedding fusion (HGAT-lite)
        
        # Choose layer type based on multi-hop setting
        if use_multihop:
            self.layers = nn.ModuleList([
                MultiHopEncoderLayer(d_model, n_heads, d_ff, num_relations, max_hops, dropout)
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                TinyEncoderLayer(d_model, n_heads, d_ff, num_relations, dropout, use_rel_bias=use_rel_attention_bias)
                for _ in range(n_layers)
            ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, rel_ids: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        
        # Token embeddings
        x = self.tok_emb(input_ids)
        
        # Graph-aware positional encoding
        if rel_ids is not None:
            pos_emb = self.graph_pos_encoding(input_ids, rel_ids)
            x = x + pos_emb
        else:
            # Fallback to standard positional encoding if no relation IDs
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            # Create dummy pos_emb for compatibility
            pos_emb = torch.zeros_like(x)
            x = x + pos_emb
        
        # Relation embeddings (HGAT-lite fusion)
        if rel_ids is not None:
            x = x + self.rel_emb(rel_ids)
        
        # Build key padding mask for attention
        if attention_mask is not None:
            kpm = attention_mask  # B,T where 1 keep
        else:
            kpm = None
        
        # Disable relation bias in attention when flag is off by not forwarding rel_ids
        if self.use_multihop:
            # Multi-hop layers always use relation IDs
            rel_for_attn = rel_ids
        else:
            # Standard layers respect the bias flag
            rel_for_attn = rel_ids if self.use_rel_attention_bias else None
        
        for layer in self.layers:
            x = layer(x, attn_mask=kpm, rel_ids=rel_for_attn)
        
        x = self.norm(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


class MNMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.node_head = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.node_head(hidden)

"""
Graph Positional Encoding - Preserves graph structure in transformer attention.
Extends standard positional encoding with graph-aware position embeddings.
"""
import torch
import torch.nn as nn
import math
from typing import Dict, List, Tuple


class GraphPositionalEncoding(nn.Module):
    """Graph-aware positional encoding that preserves KG structure."""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, max_graph_size: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Standard positional encoding for sequence positions
        self.seq_pos_encoding = nn.Embedding(max_seq_len, d_model)
        
        # Graph-specific positional encodings
        self.chain_pos_encoding = nn.Embedding(max_graph_size, d_model)  # Position within chain
        self.depth_pos_encoding = nn.Embedding(20, d_model)  # Graph depth (max 20 levels)
        self.role_pos_encoding = nn.Embedding(4, d_model)  # Entity/Relation/Code/Special roles
        
        # Initialize with sinusoidal patterns
        self._init_positional_encodings()
    
    def _init_positional_encodings(self):
        """Initialize with sinusoidal positional encoding patterns."""
        # Standard sequence positions
        seq_pos = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        seq_pos[:, 0::2] = torch.sin(position * div_term)
        seq_pos[:, 1::2] = torch.cos(position * div_term)
        self.seq_pos_encoding.weight.data = seq_pos
        
        # Chain positions (similar pattern but different frequency)
        chain_pos = torch.zeros(1000, self.d_model)
        chain_position = torch.arange(0, 1000).unsqueeze(1).float()
        chain_div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                                 -(math.log(5000.0) / self.d_model))
        
        chain_pos[:, 0::2] = torch.sin(chain_position * chain_div_term)
        chain_pos[:, 1::2] = torch.cos(chain_position * chain_div_term)
        self.chain_pos_encoding.weight.data = chain_pos
        
        # Initialize others with small random values
        nn.init.normal_(self.depth_pos_encoding.weight, std=0.02)
        nn.init.normal_(self.role_pos_encoding.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Add graph-aware positional encodings to input embeddings.
        
        Args:
            input_ids: Token IDs [B, T]
            rel_ids: Relation IDs for graph structure [B, T]
            
        Returns:
            Positional encodings [B, T, D]
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Standard sequence positions
        seq_positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        seq_pos_emb = self.seq_pos_encoding(seq_positions)
        
        # Graph-aware positions
        chain_positions, depths, roles = self._compute_graph_positions(input_ids, rel_ids)
        
        chain_pos_emb = self.chain_pos_encoding(chain_positions)
        depth_pos_emb = self.depth_pos_encoding(depths)
        role_pos_emb = self.role_pos_encoding(roles)
        
        # Combine all positional encodings
        graph_pos_emb = seq_pos_emb + 0.5 * chain_pos_emb + 0.3 * depth_pos_emb + 0.2 * role_pos_emb
        
        return graph_pos_emb
    
    def _compute_graph_positions(self, input_ids: torch.Tensor, rel_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute graph-specific position information."""
        B, T = input_ids.shape
        device = input_ids.device
        
        # Initialize position tensors
        chain_positions = torch.zeros(B, T, dtype=torch.long, device=device)
        depths = torch.zeros(B, T, dtype=torch.long, device=device)
        roles = torch.zeros(B, T, dtype=torch.long, device=device)
        
        # Special token IDs (from dataset)
        SPECIAL_TOKENS = {0, 1, 2, 3, 4}  # PAD, CLS, SEP, MASK, UNK
        
        for b in range(B):
            chain_pos = 0
            current_depth = 0
            
            for t in range(T):
                token_id = input_ids[b, t].item()
                rel_id = rel_ids[b, t].item()
                
                # Determine role
                if token_id in SPECIAL_TOKENS:
                    role = 3  # Special token
                elif rel_id == 0:
                    role = 2  # Code token
                elif self._is_relation_token(token_id):
                    role = 1  # Relation token
                    current_depth += 1
                else:
                    role = 0  # Entity token
                
                roles[b, t] = role
                
                # Chain position (increments within each relation chain)
                if rel_id > 0:
                    chain_positions[b, t] = chain_pos
                    chain_pos += 1
                else:
                    if rel_id == 0 and t > 0 and rel_ids[b, t-1] > 0:
                        # End of chain, reset position
                        chain_pos = 0
                    chain_positions[b, t] = chain_pos
                
                # Depth (based on relation nesting)
                depths[b, t] = min(current_depth, 19)  # Cap at max depth
        
        return chain_positions, depths, roles
    
    def _is_relation_token(self, token_id: int) -> bool:
        """Check if token represents a relation (heuristic)."""
        # This is a simplified heuristic - in practice, you'd maintain
        # a proper mapping of relation tokens
        return False  # Simplified for now


class GraphAwareEncoder(nn.Module):
    """Enhanced encoder with graph positional encoding."""
    
    def __init__(self, d_model: int, vocab_size: int, max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Graph positional encoding
        self.graph_pos_encoding = GraphPositionalEncoding(d_model, max_seq_len)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with graph-aware positional encoding.
        
        Args:
            input_ids: Token IDs [B, T]
            rel_ids: Relation IDs [B, T]
            
        Returns:
            Embeddings with graph positional encoding [B, T, D]
        """
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Graph positional encodings
        pos_emb = self.graph_pos_encoding(input_ids, rel_ids)
        
        # Combine and normalize
        embeddings = token_emb + pos_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


def test_graph_positional():
    """Test graph positional encoding."""
    print("Testing Graph Positional Encoding...")
    
    # Create test data
    B, T, D = 2, 10, 768
    input_ids = torch.randint(0, 1000, (B, T))
    rel_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 0, 0],
                           [0, 0, 1, 1, 0, 2, 2, 2, 2, 0]])
    
    # Test positional encoding
    pos_encoder = GraphPositionalEncoding(D)
    pos_emb = pos_encoder(input_ids, rel_ids)
    
    print(f"✅ Positional encoding shape: {pos_emb.shape}")
    print(f"✅ Output range: [{pos_emb.min():.3f}, {pos_emb.max():.3f}]")
    
    # Test full encoder
    encoder = GraphAwareEncoder(D, vocab_size=1000)
    output = encoder(input_ids, rel_ids)
    
    print(f"✅ Full encoder output shape: {output.shape}")
    print("✅ Graph positional encoding test passed!")


if __name__ == "__main__":
    test_graph_positional()

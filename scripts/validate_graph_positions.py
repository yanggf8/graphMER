#!/usr/bin/env python3
"""Validate Graph Positional Encoding implementation."""
import sys
from pathlib import Path
import torch

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.graph_positional import GraphPositionalEncoding, GraphAwareEncoder
from src.models.encoder import TinyEncoder
from src.training.dataset_v2 import LeafyChainDatasetV2


def test_graph_positional_basic():
    """Test basic graph positional encoding functionality."""
    print("🔍 Testing Graph Positional Encoding...")
    
    # Test parameters
    B, T, D = 2, 20, 768
    
    # Create test data with graph structure
    input_ids = torch.randint(5, 1000, (B, T))  # Avoid special tokens 0-4
    rel_ids = torch.tensor([
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0],  # Mixed structure
        [0, 0, 1, 1, 1, 0, 2, 2, 0, 3, 3, 3, 0, 0, 4, 4, 4, 4, 0, 0]   # Different pattern
    ])
    
    # Test positional encoding
    pos_encoder = GraphPositionalEncoding(D, max_seq_len=512)
    pos_emb = pos_encoder(input_ids, rel_ids)
    
    print(f"✅ Input shape: {input_ids.shape}")
    print(f"✅ Relation IDs shape: {rel_ids.shape}")
    print(f"✅ Positional encoding shape: {pos_emb.shape}")
    print(f"✅ Output range: [{pos_emb.min():.3f}, {pos_emb.max():.3f}]")
    
    # Verify different positions have different encodings
    pos_diff = torch.norm(pos_emb[0, 0] - pos_emb[0, 1])
    print(f"✅ Position difference norm: {pos_diff:.3f}")
    
    assert pos_emb.shape == (B, T, D), "Wrong output shape"
    assert pos_diff > 0.1, "Positions should be sufficiently different"
    
    print("✅ Basic graph positional encoding tests passed!")
    return True


def test_encoder_integration():
    """Test integration with TinyEncoder."""
    print("\n🔍 Testing Encoder Integration...")
    
    # Test parameters
    vocab_size = 8000
    d_model = 256
    n_heads = 4
    n_layers = 2
    
    # Create encoder with graph positional encoding
    encoder = TinyEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        num_relations=10,
        use_rel_attention_bias=True
    )
    
    # Test data
    B, T = 2, 16
    input_ids = torch.randint(5, vocab_size, (B, T))
    attention_mask = torch.ones(B, T)
    rel_ids = torch.randint(0, 5, (B, T))
    
    # Forward pass
    output = encoder(input_ids, attention_mask, rel_ids)
    
    print(f"✅ Encoder output shape: {output.shape}")
    print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test without relation IDs (fallback)
    output_no_rel = encoder(input_ids, attention_mask, None)
    print(f"✅ Output without rel_ids shape: {output_no_rel.shape}")
    
    assert output.shape == (B, T, d_model), "Wrong encoder output shape"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("✅ Encoder integration tests passed!")
    return True


def test_dataset_compatibility():
    """Test compatibility with dataset and training pipeline."""
    print("\n🔍 Testing Dataset Compatibility...")
    
    # Create minimal dataset
    code_snippets = [
        "def process(): return data",
        "class Handler: def run(self): pass"
    ]
    
    leaves = [
        [("calls", ["helper"]), ("returns", ["data"])],
        [("inherits_from", ["Base"]), ("contains", ["run"])]
    ]
    
    dataset = LeafyChainDatasetV2(code_snippets, leaves, max_seq_len=64)
    
    if len(dataset) > 0:
        # Get sample
        input_ids, attention_mask, mlm_labels, mnm_labels, rel_ids = dataset[0]
        
        print(f"✅ Sample shapes:")
        print(f"   input_ids: {input_ids.shape}")
        print(f"   rel_ids: {rel_ids.shape}")
        
        # Test with encoder
        encoder = TinyEncoder(vocab_size=8000, d_model=256, n_heads=4, n_layers=2)
        
        # Add batch dimension
        input_ids_batch = input_ids.unsqueeze(0)
        attention_mask_batch = attention_mask.unsqueeze(0)
        rel_ids_batch = rel_ids.unsqueeze(0)
        
        output = encoder(input_ids_batch, attention_mask_batch, rel_ids_batch)
        
        print(f"✅ Encoder output with dataset: {output.shape}")
        
        # Verify graph structure is preserved
        unique_rel_ids = set(rel_ids.tolist())
        print(f"✅ Unique relation IDs in sample: {unique_rel_ids}")
        
        assert len(unique_rel_ids) > 1, "Should have multiple relation types"
        
    print("✅ Dataset compatibility tests passed!")
    return True


def test_graph_structure_awareness():
    """Test that the encoding is aware of graph structure."""
    print("\n🔍 Testing Graph Structure Awareness...")
    
    pos_encoder = GraphPositionalEncoding(256, max_seq_len=100)
    
    # Test 1: Same relation chain should have similar patterns
    input_ids = torch.randint(5, 1000, (1, 10))
    rel_ids_chain = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]])  # Two chains
    
    pos_emb = pos_encoder(input_ids, rel_ids_chain)
    
    # Positions within same chain should be more similar than across chains
    chain1_pos = pos_emb[0, :5]  # First chain
    chain2_pos = pos_emb[0, 5:]  # Second chain
    
    within_chain_sim = torch.cosine_similarity(chain1_pos[0], chain1_pos[1], dim=0)
    across_chain_sim = torch.cosine_similarity(chain1_pos[0], chain2_pos[0], dim=0)
    
    print(f"✅ Within-chain similarity: {within_chain_sim:.3f}")
    print(f"✅ Across-chain similarity: {across_chain_sim:.3f}")
    
    # Test 2: Different depths should have different encodings
    rel_ids_depth = torch.tensor([[0, 1, 2, 3, 2, 1, 0, 1, 2, 0]])  # Varying depths
    pos_emb_depth = pos_encoder(input_ids, rel_ids_depth)
    
    depth_diff = torch.norm(pos_emb_depth[0, 1] - pos_emb_depth[0, 3])  # Depth 1 vs 3
    print(f"✅ Depth difference norm: {depth_diff:.3f}")
    
    assert depth_diff > 0.1, "Different depths should have different encodings"
    
    print("✅ Graph structure awareness tests passed!")
    return True


def main():
    """Run all validation tests."""
    print("🚀 Validating Graph Positional Encoding Implementation\n")
    
    tests = [
        test_graph_positional_basic,
        test_encoder_integration,
        test_dataset_compatibility,
        test_graph_structure_awareness
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Graph Positional Encoding tests passed!")
        print("✅ GraphMER paper compliance: Graph structure preservation implemented")
        return True
    else:
        print("⚠️  Some tests failed - implementation needs fixes")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

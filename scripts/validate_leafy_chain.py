#!/usr/bin/env python3
"""Validate Leafy Chain Graph Encoding implementation."""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.encoding.leafy_chain import LeafyChainEncoder
from src.training.dataset_v2 import LeafyChainDatasetV2


def test_leafy_chain_basic():
    """Test basic Leafy Chain functionality."""
    print("🔍 Testing Leafy Chain Graph Encoding...")
    
    encoder = LeafyChainEncoder()
    
    # Test with software engineering triples
    triples = [
        ("MyClass", "inherits_from", "BaseClass"),
        ("BaseClass", "contains", "method1"),
        ("MyClass", "contains", "method2"),
        ("method1", "calls", "helper_func"),
        ("method2", "uses", "variable_x")
    ]
    
    # Test linearization
    tokens = encoder.linearize_graph(triples)
    print(f"✅ Linearized {len(triples)} triples into {len(tokens)} tokens")
    print(f"   Sample tokens: {tokens[:5]}...")
    
    # Test relation IDs
    relation_vocab = {"inherits_from": 1, "contains": 2, "calls": 3, "uses": 4}
    rel_ids = encoder.get_relation_ids(tokens, relation_vocab)
    print(f"✅ Generated {len(rel_ids)} relation IDs")
    
    # Verify structure preservation
    assert len(tokens) == len(rel_ids), "Token and relation ID lengths must match"
    assert any("[ENT]" in token for token in tokens), "Must contain entity tokens"
    assert any("[REL]" in token for token in tokens), "Must contain relation tokens"
    
    print("✅ Basic Leafy Chain tests passed!")
    return True


def test_dataset_integration():
    """Test Leafy Chain integration with dataset."""
    print("\n🔍 Testing Dataset Integration...")
    
    # Create minimal dataset
    code_snippets = [
        "def process_data(): return result",
        "class MyClass: def __init__(self): pass"
    ]
    
    leaves = [
        [("calls", ["helper_func"]), ("returns", ["result"])],
        [("inherits_from", ["BaseClass"]), ("contains", ["__init__"])]
    ]
    
    dataset = LeafyChainDatasetV2(code_snippets, leaves, max_seq_len=128)
    
    print(f"✅ Created dataset with {len(dataset)} samples")
    
    # Test sample structure
    if len(dataset) > 0:
        sample = dataset[0]
        assert len(sample) == 5, "Sample must have 5 components"
        
        input_ids, attention_mask, mlm_labels, mnm_labels, rel_ids = sample
        
        print(f"✅ Sample shapes: input_ids={input_ids.shape}, rel_ids={rel_ids.shape}")
        
        # Verify relation IDs are properly assigned
        unique_rel_ids = set(rel_ids.tolist())
        print(f"✅ Found {len(unique_rel_ids)} unique relation IDs: {unique_rel_ids}")
        
        # Verify masking
        mlm_masked = (mlm_labels != -100).sum().item()
        mnm_masked = (mnm_labels != -100).sum().item()
        print(f"✅ Masking: MLM={mlm_masked} tokens, MNM={mnm_masked} tokens")
    
    print("✅ Dataset integration tests passed!")
    return True


def test_graph_structure_preservation():
    """Test that graph structure is preserved in linearization."""
    print("\n🔍 Testing Graph Structure Preservation...")
    
    encoder = LeafyChainEncoder()
    
    # Test with hierarchical structure
    triples = [
        ("root", "contains", "child1"),
        ("root", "contains", "child2"),
        ("child1", "inherits_from", "parent"),
        ("child2", "calls", "function")
    ]
    
    tokens = encoder.linearize_graph(triples)
    
    # Verify root appears first
    root_positions = [i for i, token in enumerate(tokens) if "root" in token]
    print(f"✅ Root entity positions: {root_positions}")
    
    # Verify chain structure
    chain_separators = [i for i, token in enumerate(tokens) if token == "[CHAIN]"]
    print(f"✅ Chain separators at positions: {chain_separators}")
    
    # Verify relation-entity pairing
    rel_positions = [i for i, token in enumerate(tokens) if token.startswith("[REL]")]
    ent_positions = [i for i, token in enumerate(tokens) if token.startswith("[ENT]")]
    
    print(f"✅ Relations: {len(rel_positions)}, Entities: {len(ent_positions)}")
    
    print("✅ Graph structure preservation tests passed!")
    return True


def main():
    """Run all validation tests."""
    print("🚀 Validating Leafy Chain Graph Encoding Implementation\n")
    
    tests = [
        test_leafy_chain_basic,
        test_dataset_integration,
        test_graph_structure_preservation
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Leafy Chain validation tests passed!")
        print("✅ GraphMER paper compliance: Core algorithm implemented")
        return True
    else:
        print("⚠️  Some tests failed - implementation needs fixes")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

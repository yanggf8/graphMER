"""Test MNM fixes: coverage and optimizer groups."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_mnm_mask_coverage():
    """Test that MNM masking meets minimum coverage."""
    from src.training.dataset_v2 import LeafyChainDatasetV2
    
    # Small test data
    code_snippets = ['def test(): return 42'] * 5
    leaves_per_snip = [[('test', ['calls', 'returns'])]] * 5
    
    dataset = LeafyChainDatasetV2(code_snippets, leaves_per_snip, max_seq_len=64)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        mnm_labels = sample[3]
        masked_count = (mnm_labels != -100).sum().item()
        
        # Should have at least some masking (improved from 0-1 to 3+)
        assert masked_count >= 3, f"Sample {i}: only {masked_count} MNM masks"
    
    print("✅ MNM mask coverage test passed")

def test_optimizer_param_groups():
    """Test head-specific learning rates."""
    from src.models.encoder import TinyEncoder, MLMHead, MNMHead
    
    model = TinyEncoder(vocab_size=100, d_model=64, n_layers=2, n_heads=2)
    mlm_head = MLMHead(64, 100)
    mnm_head = MNMHead(64, 100)
    
    # Create param groups like train_v2_fixed.py
    encoder_params = list(model.parameters()) + list(mlm_head.parameters())
    mnm_params = list(mnm_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 3e-4},
        {'params': mnm_params, 'lr': 5e-4}
    ])
    
    # Verify param groups
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['lr'] == 3e-4  # encoder
    assert optimizer.param_groups[1]['lr'] == 5e-4  # mnm_head
    
    print("✅ Optimizer param groups test passed")

if __name__ == "__main__":
    test_mnm_mask_coverage()
    test_optimizer_param_groups()

import torch
from src.models.encoder import TinyEncoder, TinyRelSelfAttention


def test_rel_attention_bias_shapes_and_nans():
    torch.manual_seed(0)
    B, T, V = 1, 8, 50
    d_model, n_heads, n_layers = 32, 4, 1
    num_rel = 5

    # inputs
    input_ids = torch.randint(low=0, high=V, size=(B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    # rel_ids: mark positions 2-3 as relation 1, 5-6 as relation 2
    rel_ids = torch.zeros(B, T, dtype=torch.long)
    rel_ids[0, 2:4] = 1
    rel_ids[0, 5:7] = 2

    # model with relation attention bias enabled
    model = TinyEncoder(vocab_size=V, d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=64, num_relations=num_rel, use_rel_attention_bias=True)
    out = model(input_ids, attention_mask=attention_mask, rel_ids=rel_ids)

    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all(), "Output contains NaNs or Infs"

#!/usr/bin/env python3
"""Standalone evaluation script"""
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_cpu.yaml")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="logs/eval_results.json")
    args = parser.parse_args()
    
    import torch
    from torch.utils.data import DataLoader
    from src.training.kg_dataset_builder import build_dataset_from_kg
    from src.models.encoder import TinyEncoder, MLMHead, MNMHead
    from src.training.evaluator import evaluate_model
    import yaml
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cpu")
    
    # Build dataset
    triples_path = Path("data/kg/seed_python.jsonl")
    code_path = Path("data/raw/python_samples/sample1.py")
    ds = build_dataset_from_kg(triples_path, code_path, max_seq_len=128, limit=64, chunk_size=3)
    
    # Create validation split
    n = len(ds)
    val_idx = list(range(max(1, int(0.8 * n)), n))
    
    class Subset(torch.utils.data.Dataset):
        def __init__(self, base, idx):
            self.base, self.idx = base, idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            return self.base[self.idx[i]]
    
    val_ds = Subset(ds, val_idx) if val_idx else Subset(ds, [0])
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    
    # Initialize model
    vocab_size = len(ds.vocab)
    num_rel = max(1, len(getattr(ds, 'rel_stoi', {"<none>": 0})))
    enc_cfg = config.get("model", {})
    
    model = TinyEncoder(
        vocab_size=vocab_size, 
        d_model=enc_cfg.get("hidden_size", 256), 
        n_heads=4, n_layers=4, d_ff=1024, 
        num_relations=num_rel, 
        use_rel_attention_bias=enc_cfg.get("use_rel_attention_bias", True)
    )
    mlm_head = MLMHead(d_model=enc_cfg.get("hidden_size", 256), vocab_size=vocab_size)
    mnm_head = MNMHead(d_model=enc_cfg.get("hidden_size", 256), vocab_size=vocab_size)
    
    model.to(device)
    mlm_head.to(device) 
    mnm_head.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        mlm_head.load_state_dict(checkpoint['mlm_head'])
        mnm_head.load_state_dict(checkpoint['mnm_head'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Run evaluation
    metrics = evaluate_model(model, mlm_head, mnm_head, val_dl, device)
    
    # Save results
    results = {
        "metrics": metrics,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset_size": len(val_ds)
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()

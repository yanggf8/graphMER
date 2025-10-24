#!/usr/bin/env python3
"""Updated training script that uses fixed dataset with proper BPE tokenizer integration."""
from pathlib import Path
import argparse
import sys
import yaml
import os
from datetime import datetime

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_cpu.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to use (None = all)")
    parser.add_argument("--use_full_kg", action="store_true", help="Use full KG dataset (all 29k triples)")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Per-step micro batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save checkpoint every N steps (0=only final)")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    print("Loaded training config keys:", list(config.keys()))

    import torch
    # Set random seeds for reproducibility
    import random
    import numpy as np
    seed = args.seed or config.get("run", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Using random seed: {seed}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from torch.utils.data import DataLoader
    from src.training.dataset_v2 import LeafyChainDatasetV2
    from src.training.kg_dataset_builder_v2 import build_dataset_from_kg_simple, build_dataset_from_kg_full
    from src.models.encoder import TinyEncoder, MLMHead, MNMHead
    from src.training.metrics import masked_token_accuracy

    # Build dataset from real KG
    # Prefer seed_python.jsonl; fallback to enhanced_multilang.jsonl if present
    candidates = [
        Path("data/kg/seed_python.jsonl"),
        Path("data/kg/enhanced_multilang.jsonl"),
    ]
    triples_path = next((p for p in candidates if p.exists()), None)
    
    if triples_path is None:
        print("Error: No triples file found. Expected one of:")
        for p in candidates:
            print(" -", p)
        sys.exit(1)
    else:
        print(f"Using triples file: {triples_path}")
    
    if args.use_full_kg:
        print("Building dataset from FULL KG (all triples)...")
        # Get all code files
        code_dir = Path("data/raw/python_samples")
        code_paths = list(code_dir.glob("*.py"))
        code_paths.extend(code_dir.glob("**/*.py"))
        
        ds = build_dataset_from_kg_full(
            triples_path, 
            code_paths[:50],  # Use first 50 files for now
            max_seq_len=128,
            max_samples=args.max_samples,
            max_leaves_per_sample=5
        )
    else:
        print("Building dataset from KG (simple mode)...")
        code_path = Path("data/raw/python_samples/sample1.py")
        ds = build_dataset_from_kg_simple(
            triples_path, 
            code_path, 
            max_seq_len=128,
            limit=args.max_samples or 1000,
            chunk_size=5
        )
    
    vocab_size = ds.vocab_size
    print(f"✅ Using BPE vocabulary with {vocab_size} tokens")
    
    # Split into train/val
    n = len(ds)
    n_train = max(1, int(0.8 * n))
    indices = list(range(n))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:] if n_train < n else [n_train - 1]
    
    class Subset(torch.utils.data.Dataset):
        def __init__(self, base, idx):
            self.base, self.idx = base, idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            return self.base[self.idx[i]]
    
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # DataLoaders
    dl = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True, collate_fn=lambda b: b[0])
    val_dl = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=lambda b: b[0])

    # Model
    enc_cfg = config.get("model", {})
    num_rel = len(ds.rel_stoi)
    
    hidden_size = enc_cfg.get("hidden_size", 768)
    num_layers = enc_cfg.get("num_layers", 12)
    num_heads = enc_cfg.get("num_heads", 12)
    intermediate_size = enc_cfg.get("intermediate_size", 3072)
    use_rel_bias = bool(enc_cfg.get("use_rel_attention_bias", True))
    
    print(f"\nModel Architecture:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num layers: {num_layers}")
    print(f"  Num heads: {num_heads}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Num relations: {num_rel}")
    print(f"  Relation attention bias: {use_rel_bias}")
    
    model = TinyEncoder(
        vocab_size=vocab_size, 
        d_model=hidden_size, 
        n_heads=num_heads, 
        n_layers=num_layers, 
        d_ff=intermediate_size, 
        num_relations=num_rel, 
        use_rel_attention_bias=use_rel_bias
    )
    mlm_head = MLMHead(d_model=hidden_size, vocab_size=vocab_size)
    mnm_head = MNMHead(d_model=hidden_size, vocab_size=vocab_size)
    
    model.to(device)
    mlm_head.to(device)
    mnm_head.to(device)

    opt_cfg = config.get("optimizer", {})
    optim = torch.optim.AdamW(
        list(model.parameters()) + list(mlm_head.parameters()) + list(mnm_head.parameters()), 
        lr=opt_cfg.get("lr", 3e-4), 
        weight_decay=opt_cfg.get("weight_decay", 0.01)
    )
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))
    steps = int(args.steps)
    model.train()
    
    import csv
    log_path = Path("logs/train_v2_metrics.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting training for {steps} steps...")
    print(f"Logging to: {log_path}")
    
    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["step", "total_loss", "mlm_loss", "mnm_loss", "mlm_validation_accuracy", "mnm_validation_accuracy"])
        
        dl_iter = iter(dl)
        val_iter = iter(val_dl)
        
        accum = 0
        optim.zero_grad(set_to_none=True)
        for step in range(steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)
            
            input_ids, attn, mlm_labels, mnm_labels, rel_ids = [x.to(device) for x in batch]
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                hidden = model(input_ids.unsqueeze(0), attn.unsqueeze(0), rel_ids.unsqueeze(0))
                logits_mlm = mlm_head(hidden)
                logits_mnm = mnm_head(hidden)
                loss_mlm = loss_fct(logits_mlm.view(-1, vocab_size), mlm_labels.view(-1))
                loss_mnm = loss_fct(logits_mnm.view(-1, vocab_size), mnm_labels.view(-1))
                loss = (loss_mlm + loss_mnm) / max(1, args.grad_accum_steps)
            scaler.scale(loss).backward()
            accum += 1
            if accum % args.grad_accum_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
            
            val_acc_mlm = 0.0
            val_acc_mnm = 0.0
            
            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{steps}: loss={(loss.item()*max(1, args.grad_accum_steps)):.4f}, mlm_loss={loss_mlm.item():.4f}, mnm_loss={loss_mnm.item():.4f}")
                
                # Validation
                with torch.no_grad():
                    try:
                        vbatch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dl)
                        vbatch = next(val_iter)
                    
                    vi, va, vml, vmn, vr = [x.to(device) for x in vbatch]
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        vhidden = model(vi.unsqueeze(0), va.unsqueeze(0), vr.unsqueeze(0))
                        vlogits_mlm = mlm_head(vhidden)
                        vlogits_mnm = mnm_head(vhidden)
                    val_acc_mlm = masked_token_accuracy(vlogits_mlm, vml.unsqueeze(0))
                    val_acc_mnm = masked_token_accuracy(vlogits_mnm, vmn.unsqueeze(0))
                    print(f"  Val: mlm_acc={val_acc_mlm:.4f}, mnm_acc={val_acc_mnm:.4f}")
            
            # Intermediate checkpointing
            if args.save_every_steps and (step + 1) % int(args.save_every_steps) == 0:
                checkpoint_dir = Path("logs/checkpoints")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                intermediate_path = checkpoint_dir / f"model_v2_step{step+1}_{timestamp}_s{args.seed}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'mlm_head_state_dict': mlm_head.state_dict(),
                    'mnm_head_state_dict': mnm_head.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'config': config,
                    'step': step + 1,
                    'loss': loss.item() * max(1, args.grad_accum_steps)
                }, intermediate_path)
                print(f"  Saved intermediate checkpoint: {intermediate_path}")
            
            writer.writerow([
                step+1, 
                float(loss.item()*max(1, args.grad_accum_steps)), 
                float(loss_mlm.item()), 
                float(loss_mnm.item()), 
                float(val_acc_mlm), 
                float(val_acc_mnm)
            ])

    # Save checkpoint
    checkpoint_dir = Path("logs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"model_v2_{timestamp}_s{args.seed}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'mlm_head_state_dict': mlm_head.state_dict(),
        'mnm_head_state_dict': mnm_head.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'config': config,
        'steps': steps,
        'seed': seed,
        'vocab_size': vocab_size,
    }, checkpoint_path)
    print(f"\n✅ Model checkpoint saved to {checkpoint_path}")
    print(f"✅ Training complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Training with MNM fixes: better masking + head-specific LR."""
import sys
from pathlib import Path
import argparse
import yaml
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_cpu.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--mnm_weight", type=float, default=0.2)
    parser.add_argument("--mnm_head_lr", type=float, default=5e-4)
    parser.add_argument("--encoder_lr", type=float, default=3e-4)
    parser.add_argument("--freeze_encoder_steps", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    
    from src.training.kg_dataset_builder_v2 import build_dataset_from_kg_full
    from src.models.encoder import TinyEncoder, MLMHead, MNMHead
    from src.training.metrics import masked_token_accuracy
    from torch.utils.data import DataLoader
    import csv
    
    # Build dataset - use working approach from original train.py
    from src.training.kg_dataset_builder_v2 import build_dataset_from_kg_simple
    from pathlib import Path
    import glob
    
    kg_path = Path("data/kg/enhanced_multilang.jsonl")
    
    # Find first Python file for simple builder
    python_files = list(Path("data/raw/python_samples").glob("*.py"))
    if not python_files:
        python_files = list(Path("data/raw").rglob("*.py"))
    
    if python_files:
        code_path = python_files[0]  # Use first file found
        dataset = build_dataset_from_kg_simple(kg_path, code_path, max_seq_len=128, limit=1000)
    else:
        raise FileNotFoundError("No Python files found for dataset building")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Dataset: {len(dataset)} samples, vocab: {dataset.vocab_size}")
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyEncoder(
        vocab_size=dataset.vocab_size,
        d_model=768, n_layers=12, n_heads=12,
        d_ff=3072, num_relations=3,
        use_rel_attention_bias=True
    ).to(device)
    
    mlm_head = MLMHead(768, dataset.vocab_size).to(device)
    mnm_head = MNMHead(768, dataset.vocab_size).to(device)
    
    # Head-specific optimizer groups
    encoder_params = list(model.parameters()) + list(mlm_head.parameters())
    mnm_params = list(mnm_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr},
        {'params': mnm_params, 'lr': args.mnm_head_lr}
    ], weight_decay=0.01)
    
    # Training loop with encoder freezing
    model.train()
    mlm_head.train()
    mnm_head.train()
    
    # Best checkpoint tracking
    best_mnm_acc = 0.0
    best_checkpoint_path = None
    
    # Freeze encoder initially if requested
    if args.freeze_encoder_steps > 0:
        for param in model.parameters():
            param.requires_grad = False
        for param in mlm_head.parameters():
            param.requires_grad = False
        print(f"Encoder frozen for first {args.freeze_encoder_steps} steps")
    
    # Metrics logging
    Path("logs").mkdir(exist_ok=True)
    with open("logs/train_v2_fixed_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total_loss", "mlm_loss", "mnm_loss", "mlm_acc", "mnm_acc", "mnm_masked_count"])
        
        for step in range(args.steps):
            # Unfreeze encoder after warmup
            if step == args.freeze_encoder_steps and args.freeze_encoder_steps > 0:
                for param in model.parameters():
                    param.requires_grad = True
                for param in mlm_head.parameters():
                    param.requires_grad = True
                print(f"Encoder unfrozen at step {step}")
            
            batch = next(iter(dataloader))
            input_ids, attention_mask, mlm_labels, mnm_labels, rel_ids = [x.to(device) for x in batch]
            
            # Count MNM masked positions
            mnm_masked_count = (mnm_labels != -100).sum().item()
            
            # Forward pass
            hidden = model(input_ids, attention_mask, rel_ids)
            mlm_logits = mlm_head(hidden)
            mnm_logits = mnm_head(hidden)
            
            # Losses
            mlm_loss = torch.nn.functional.cross_entropy(
                mlm_logits.view(-1, dataset.vocab_size), 
                mlm_labels.view(-1), 
                ignore_index=-100
            )
            mnm_loss = torch.nn.functional.cross_entropy(
                mnm_logits.view(-1, dataset.vocab_size), 
                mnm_labels.view(-1), 
                ignore_index=-100
            )
            
            # Dynamic MNM weight ramp
            mnm_weight = min(args.mnm_weight * (step / 500), args.mnm_weight)
            total_loss = mlm_loss + mnm_weight * mnm_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(mlm_head.parameters()) + list(mnm_head.parameters()), 
                1.0
            )
            optimizer.step()
            
            # Metrics every 50 steps
            if step % 50 == 0:
                with torch.no_grad():
                    mlm_acc = masked_token_accuracy(mlm_logits, mlm_labels)
                    mnm_acc = masked_token_accuracy(mnm_logits, mnm_labels)
                    
                    print(f"Step {step}: loss={total_loss:.4f}, mlm_acc={mlm_acc:.3f}, mnm_acc={mnm_acc:.3f}, mnm_masks={mnm_masked_count}")
                    
                    writer.writerow([step, total_loss.item(), mlm_loss.item(), mnm_loss.item(), 
                                   mlm_acc, mnm_acc, mnm_masked_count])
                    
                    # Save best checkpoint
                    if mnm_acc > best_mnm_acc:
                        best_mnm_acc = mnm_acc
                        best_checkpoint_path = f"logs/checkpoints/best_mnm_acc_{mnm_acc:.3f}_step{step}.pt"
                        torch.save({
                            'model': model.state_dict(),
                            'mlm_head': mlm_head.state_dict(),
                            'mnm_head': mnm_head.state_dict(),
                            'step': step,
                            'mnm_acc': mnm_acc
                        }, best_checkpoint_path)
                        print(f"  💾 New best MNM accuracy: {mnm_acc:.3f}")
    
    print(f"Training completed! Best MNM accuracy: {best_mnm_acc:.3f}")
    if best_checkpoint_path:
        print(f"Best checkpoint: {best_checkpoint_path}")

if __name__ == "__main__":
    main()

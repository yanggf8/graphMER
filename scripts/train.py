#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import yaml

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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--rel_bias", type=str, choices=["true", "false", "config"], default="config", help="Override use_rel_attention_bias (true/false) or use value from config")
    parser.add_argument("--limit", type=int, default=32, help="Max samples from KG dataset builder")
    parser.add_argument("--chunk_size", type=int, default=2, help="Chunk size for KG dataset builder")
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
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For TPU, also set XLA-specific seed
    try:
        import torch_xla.core.xla_model as xm
        if 'xm' in locals() and xm is not None:
            xm.set_rng_state(seed)
    except ImportError:
        pass
    
    print(f"Using random seed: {seed}")
    
    # Update ablation metadata with current run parameters
    try:
        from scripts.update_metadata import update_ablation_metadata
        update_ablation_metadata(
            config_path=Path(args.config),
            seed=seed,
            steps=args.steps,
            device_type=config.get("hardware", {}).get("device", "cpu"),
            run_type="tpu" if config.get("hardware", {}).get("device") == "tpu" else "cpu"
        )
    except Exception as e:
        print(f"Warning: Could not update ablation metadata: {e}")

    # Add TPU support
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.distributed.parallel_loader as pl
        xla_available = True
    except ImportError:
        xla_available = False
        xm = None
    
    # Determine device based on config
    if config.get("hardware", {}).get("device") == "tpu" and xla_available:
        device = xm.xla_device()
        print(f"Using TPU device: {device}")
        # Get TPU core index for logging
        tpu_core = xm.get_ordinal() if xm else 0
    else:
        device = torch.device("cpu")
        tpu_core = 0
    
    from torch.utils.data import DataLoader
    from src.training.dataset import LeafyChainDataset
    from src.training.kg_dataset_builder import build_dataset_from_kg
    from src.models.encoder import TinyEncoder, MLMHead, MNMHead
    from src.training.metrics import masked_token_accuracy

    # Build dataset from real KG if present
    triples_path = Path("data/kg/seed_python.jsonl")
    code_path = Path("data/raw/python_samples/sample1.py")
    if triples_path.exists() and code_path.exists():
        # ensure KG exists from latest parser
        import subprocess, sys
        subprocess.run([sys.executable, "scripts/build_kg.py"], check=False)

        ds = build_dataset_from_kg(triples_path, code_path, max_seq_len=min(128, config.get("training_data", {}).get("max_seq_len", 128)), limit=int(args.limit), chunk_size=int(args.chunk_size))
        print("Using KG-backed dataset with", len(ds), "samples.")
    else:
        # fallback synthetic
        code_snips = [code_path.read_text(encoding="utf-8") if code_path.exists() else "def foo():\n  pass"]
        leaves = [[("calls", ["bar"]), ("reads_from", ["x"])]]
        ds = LeafyChainDataset(code_snips, leaves, max_seq_len=min(128, config.get("training_data", {}).get("max_seq_len", 128)))
    vocab_size = len(ds.vocab)
    print("Vocab size:", vocab_size)
    # split into train/val tiny sets
    from math import ceil
    n = len(ds)
    n_train = max(1, int(0.8 * n))
    # simple split
    indices = list(range(n))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    class Subset(torch.utils.data.Dataset):
        def __init__(self, base, idx):
            self.base, self.idx = base, idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            return self.base[self.idx[i]]
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if val_idx else Subset(ds, train_idx[-1:])
    
    # Configure dataloader based on hardware config
    num_workers = config.get("hardware", {}).get("num_workers", 0)
    dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0], num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=num_workers)

    # Model
    enc_cfg = config.get("model", {})
    num_rel = max(1, len(getattr(ds, 'rel_stoi', {"<none>":0})))
    # determine relation bias flag
    if args.rel_bias == "true":
        use_rel_bias = True
    elif args.rel_bias == "false":
        use_rel_bias = False
    else:
        use_rel_bias = bool(enc_cfg.get("use_rel_attention_bias", True))
    model = TinyEncoder(vocab_size=vocab_size, d_model=enc_cfg.get("hidden_size", 256), n_heads=4, n_layers=4, d_ff=1024, num_relations=num_rel, use_rel_attention_bias=use_rel_bias)
    mlm_head = MLMHead(d_model=enc_cfg.get("hidden_size", 256), vocab_size=vocab_size)
    mnm_head = MNMHead(d_model=enc_cfg.get("hidden_size", 256), vocab_size=vocab_size)
    model.to(device)
    mlm_head.to(device)
    mnm_head.to(device)

    opt_cfg = config.get("optimizer", {})
    optim = torch.optim.AdamW(list(model.parameters()) + list(mlm_head.parameters()) + list(mnm_head.parameters()), lr=opt_cfg.get("lr", 3e-4), weight_decay=opt_cfg.get("weight_decay", 0.01))
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    steps = int(args.steps)
    model.train()
    import csv
    log_path = Path("logs/train_metrics.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["step", "total_loss", "mlm_loss", "mnm_loss", "mlm_validation_accuracy", "mnm_validation_accuracy"])
        # Ensure we can run for the requested number of steps even with tiny datasets
        dl_iter = iter(dl)
        val_iter = iter(val_dl)
        for step in range(steps):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)
            # Handle gradient accumulation for TPU
            grad_accum_steps = config.get("run", {}).get("gradient_accumulation_steps", 1)
            input_ids, attn, mlm_labels, mnm_labels, rel_ids = [x.to(device) for x in batch]
            hidden = model(input_ids.unsqueeze(0), attn.unsqueeze(0), rel_ids.unsqueeze(0))
            logits_mlm = mlm_head(hidden)
            logits_mnm = mnm_head(hidden)
            # shift shape: (B,T,V) -> (B*T,V)
            loss_mlm = loss_fct(logits_mlm.view(-1, vocab_size), mlm_labels.view(-1))
            loss_mnm = loss_fct(logits_mnm.view(-1, vocab_size), mnm_labels.view(-1))
            loss = (loss_mlm + loss_mnm) / grad_accum_steps  # Scale loss for gradient accumulation
            
            # Handle XLA-specific optimization step
            if xm is not None and device.type == "xla":
                loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    xm.optimizer_step(optim)
                    optim.zero_grad()
            else:
                loss.backward()
                if (step + 1) % grad_accum_steps == 0:
                    optim.step()
                    optim.zero_grad()
            val_acc_mlm = 0.0
            val_acc_mnm = 0.0
            if (step + 1) % 10 == 0:
                print(f"step {step+1}: total_loss={loss.item():.4f}, mlm_loss={loss_mlm.item():.4f}, mnm_loss={loss_mnm.item():.4f}")
                # quick validation (cycle over val set as well)
                with torch.no_grad():
                    try:
                        vbatch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dl)
                        vbatch = next(val_iter)
                    vi, va, vml, vmn, vr = [x.to(device) for x in vbatch]
                    vhidden = model(vi.unsqueeze(0), va.unsqueeze(0), vr.unsqueeze(0))
                    vlogits_mlm = mlm_head(vhidden)
                    vlogits_mnm = mnm_head(vhidden)
                    val_acc_mlm = masked_token_accuracy(vlogits_mlm, vml.unsqueeze(0))
                    val_acc_mnm = masked_token_accuracy(vlogits_mnm, vmn.unsqueeze(0))
                    # TPU-specific: mark step to release computation graph
                    if xm is not None and device.type == "xla":
                        xm.mark_step()
                    print(f"val_acc: mlm_accuracy={val_acc_mlm:.4f}, mnm_accuracy={val_acc_mnm:.4f}")
            writer.writerow([step+1, float(loss.item()), float(loss_mlm.item()), float(loss_mnm.item()), float(val_acc_mlm), float(val_acc_mnm)])

    # TPU-specific: wait for all cores to finish
    if xm is not None and device.type == "xla":
        xm.wait_device_ops()
    
    # Update trends.json with hardware-specific information
    manifest_path = Path("data/kg/manifest.json")
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        total_triples = manifest.get("total_triples", 0)
        domain_range_ratio = manifest.get("validation", {}).get("domain_range_ratio", 0.0)
        build_duration = manifest.get("build_duration_sec", 0.0)
    else:
        total_triples = 0
        domain_range_ratio = 0.0
        build_duration = 0.0

    update_trends_json(config, device.type, ds, total_triples, domain_range_ratio, build_duration)
    
    print("Smoke training finished.")


def update_trends_json(config, device_type, ds, total_triples, domain_range_ratio, build_duration):
    """Update trends.json with hardware-specific run information."""
    import json
    from datetime import datetime
    
    # Read the training metrics to get actual values for trend tracking
    log_path = Path("logs/train_metrics.csv")
    val_acc_mnms = []
    losses = []
    
    if log_path.exists():
        import csv
        with open(log_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_acc_mnms.append(float(row['mnm_validation_accuracy']))
                losses.append(float(row['total_loss']))
    
    # Calculate improvement - compare early and late values
    val_acc_improvement = 0.0
    if len(val_acc_mnms) >= 2:
        early_val_acc = val_acc_mnms[0] if len(val_acc_mnms) <= 10 else val_acc_mnms[min(10, len(val_acc_mnms)//4)]
        final_val_acc = val_acc_mnms[-1]
        if early_val_acc != 0:
            val_acc_improvement = (final_val_acc - early_val_acc) / abs(early_val_acc)
        else:
            val_acc_improvement = float('inf') if final_val_acc != 0 else 0
    
    trends_path = Path("trends.json")
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "device_type": device_type,  # "cpu", "xla" (for TPU), etc.
        "hardware_config": config.get("hardware", {}),
        "run_config": config.get("run", {}),
        "val_acc_improvement": val_acc_improvement,
        "final_val_acc_mnm": val_acc_mnms[-1] if val_acc_mnms else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "total_triples": total_triples,
        "domain_range_ratio": domain_range_ratio,
        "build_duration": build_duration,
        "steps_completed": len(val_acc_mnms)
    }
    
    # Load existing trends or create empty list
    if trends_path.exists():
        with open(trends_path, 'r', encoding='utf-8') as f:
            trends = json.load(f)
    else:
        trends = []
    
    # Append new run info
    trends.append(run_info)
    
    # Write back to file
    with open(trends_path, 'w', encoding='utf-8') as f:
        json.dump(trends, f, indent=2)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import yaml
import itertools

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_cpu.yaml")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rel_bias", type=str, choices=["true", "false", "config"], default="config", help="Override use_rel_attention_bias (true/false) or use value from config")
    parser.add_argument("--limit", type=int, default=32, help="Max samples from KG dataset builder")
    parser.add_argument("--chunk_size", type=int, default=2, help="Chunk size for KG dataset builder")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    print("Loaded training config keys:", list(config.keys()))

    import torch
    from torch.utils.data import DataLoader
    from src.training.dataset import LeafyChainDataset
    from src.training.kg_dataset_builder import build_dataset_from_kg
    from src.models.encoder import TinyEncoder, MLMHead, MNMHead
    from src.training.metrics import masked_token_accuracy

    device = torch.device("cpu")

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
    dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])

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
    
    # Create infinite dataloader by cycling
    infinite_dl = itertools.cycle(dl)
    
    with log_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["step", "loss", "loss_mlm", "loss_mnm", "val_acc_mlm", "val_acc_mnm"])
        for step in range(steps):
            batch = next(infinite_dl)
            input_ids, attn, mlm_labels, mnm_labels, rel_ids = [x.to(device) for x in batch]
            hidden = model(input_ids.unsqueeze(0), attn.unsqueeze(0), rel_ids.unsqueeze(0))
            logits_mlm = mlm_head(hidden)
            logits_mnm = mnm_head(hidden)
            # shift shape: (B,T,V) -> (B*T,V)
            loss_mlm = loss_fct(logits_mlm.view(-1, vocab_size), mlm_labels.view(-1))
            loss_mnm = loss_fct(logits_mnm.view(-1, vocab_size), mnm_labels.view(-1))
            loss = loss_mlm + loss_mnm
            optim.zero_grad()
            loss.backward()
            optim.step()
            val_acc_mlm = 0.0
            val_acc_mnm = 0.0
            if (step + 1) % 10 == 0:
                print(f"step {step+1}: loss={loss.item():.4f} mlm={loss_mlm.item():.4f} mnm={loss_mnm.item():.4f}")
                # quick validation
                with torch.no_grad():
                    vbatch = next(iter(val_dl))
                    vi, va, vml, vmn, vr = [x.to(device) for x in vbatch]
                    vhidden = model(vi.unsqueeze(0), va.unsqueeze(0), vr.unsqueeze(0))
                    vlogits_mlm = mlm_head(vhidden)
                    vlogits_mnm = mnm_head(vhidden)
                    val_acc_mlm = masked_token_accuracy(vlogits_mlm, vml.unsqueeze(0))
                    val_acc_mnm = masked_token_accuracy(vlogits_mnm, vmn.unsqueeze(0))
                    print(f"val acc: mlm={val_acc_mlm:.3f}, mnm={val_acc_mnm:.3f}")
            writer.writerow([step+1, float(loss.item()), float(loss_mlm.item()), float(loss_mnm.item()), float(val_acc_mlm), float(val_acc_mnm)])

    print("Training finished.")

if __name__ == "__main__":
    main()

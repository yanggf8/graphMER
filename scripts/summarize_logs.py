#!/usr/bin/env python3
"""
Summarize and compare A/B logs from logs/train_metrics_A.csv and logs/train_metrics_B.csv.
Prints metrics at checkpoints (default: 100, 150, 200) and final row if available.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
from typing import List, Dict

FIELDS = ["step", "loss", "loss_mlm", "loss_mnm", "val_acc_mlm", "val_acc_mnm"]


def read_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out = {}
            for k in FIELDS:
                if k not in row:
                    continue
                if k == "step":
                    out[k] = int(float(row[k]))
                else:
                    try:
                        out[k] = float(row[k])
                    except ValueError:
                        out[k] = float("nan")
            rows.append(out)
    return rows


def pick_steps(rows: List[Dict[str, float]], steps: List[int]) -> Dict[int, Dict[str, float]]:
    idx = {row["step"]: row for row in rows}
    return {s: idx.get(s, rows[-1] if rows else {}) for s in steps}


def summarize(a_rows: List[Dict[str, float]], b_rows: List[Dict[str, float]], steps: List[int]):
    def fmt(x: float) -> str:
        if x != x:
            return "nan"
        return f"{x:.4f}"

    def rel_improve(a: float, b: float, lower_is_better: bool = True) -> str:
        if any([(a != a), (b != b), a == 0]):
            return "nan"
        delta = (a - b) / abs(a) if lower_is_better else (b - a) / abs(a)
        return f"{delta*100:.2f}%"

    if not a_rows or not b_rows:
        print("Missing logs for A or B.")
        print("To regenerate ablation logs:")
        print("  python scripts/run_ablation_fixed.py --config configs/train_cpu.yaml --steps 200")
        print("Expected files:")
        print("  - logs/train_metrics_A.csv (embedding-only)")
        print("  - logs/train_metrics_B.csv (attention-bias)")
        return

    a_last = a_rows[-1]
    b_last = b_rows[-1]

    print("Summary (A: embedding-only, B: attention-bias)")
    print("")
    print("Final step:")
    print(f"  A step={a_last.get('step')} loss={fmt(a_last.get('loss', float('nan')))} loss_mlm={fmt(a_last.get('loss_mlm', float('nan')))} loss_mnm={fmt(a_last.get('loss_mnm', float('nan')))} val_acc_mlm={fmt(a_last.get('val_acc_mlm', float('nan')))} val_acc_mnm={fmt(a_last.get('val_acc_mnm', float('nan')))}")
    print(f"  B step={b_last.get('step')} loss={fmt(b_last.get('loss', float('nan')))} loss_mlm={fmt(b_last.get('loss_mlm', float('nan')))} loss_mnm={fmt(b_last.get('loss_mnm', float('nan')))} val_acc_mlm={fmt(b_last.get('val_acc_mlm', float('nan')))} val_acc_mnm={fmt(b_last.get('val_acc_mnm', float('nan')))}")
    print(f"  Relative improvement (MNM loss, lower is better): {rel_improve(a_last.get('loss_mnm', float('nan')), b_last.get('loss_mnm', float('nan')), lower_is_better=True)}")
    print(f"  Relative improvement (MNM val_acc, higher is better): {rel_improve(a_last.get('val_acc_mnm', float('nan')), b_last.get('val_acc_mnm', float('nan')), lower_is_better=False)}")

    picks_a = pick_steps(a_rows, steps)
    picks_b = pick_steps(b_rows, steps)
    print("")
    print("Checkpoints:")
    for s in steps:
        ra = picks_a.get(s, {})
        rb = picks_b.get(s, {})
        if not ra or not rb:
            continue
        print(f"  Step {s}:")
        print(f"    A loss_mnm={fmt(ra.get('loss_mnm', float('nan')))} val_acc_mnm={fmt(ra.get('val_acc_mnm', float('nan')))} | B loss_mnm={fmt(rb.get('loss_mnm', float('nan')))} val_acc_mnm={fmt(rb.get('val_acc_mnm', float('nan')))}")
        print(f"    Δ rel (loss_mnm): {rel_improve(ra.get('loss_mnm', float('nan')), rb.get('loss_mnm', float('nan')), lower_is_better=True)} | Δ rel (val_acc_mnm): {rel_improve(ra.get('val_acc_mnm', float('nan')), rb.get('val_acc_mnm', float('nan')), lower_is_better=False)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="*", default=[100, 150, 200])
    parser.add_argument("--logA", type=str, default=str(Path("logs") / "train_metrics_A.csv"))
    parser.add_argument("--logB", type=str, default=str(Path("logs") / "train_metrics_B.csv"))
    args = parser.parse_args()

    a_rows = read_csv(Path(args.logA))
    b_rows = read_csv(Path(args.logB))
    summarize(a_rows, b_rows, args.steps)


if __name__ == "__main__":
    main()

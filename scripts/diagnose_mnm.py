#!/usr/bin/env python3
"""Diagnose MNM accuracy issues."""
import sys
from pathlib import Path
import torch
import csv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def check_recent_training_logs():
    """Check recent training logs for MNM patterns."""
    csv_path = Path("logs/train_v2_metrics.csv")
    if not csv_path.exists():
        print("No training metrics found")
        return
    
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    
    print(f"Training steps: {len(rows)}")
    
    # Check MNM accuracy progression
    mnm_accs = [float(row['mnm_validation_accuracy']) for row in rows[-10:]]
    mlm_accs = [float(row['mlm_validation_accuracy']) for row in rows[-10:]]
    
    print(f"Last 10 MNM accuracies: {mnm_accs}")
    print(f"Last 10 MLM accuracies: {mlm_accs}")
    
    # Check if MNM is always 0
    if all(acc == 0.0 for acc in mnm_accs):
        print("🚨 MNM accuracy stuck at 0% - likely metric or data issue")
    
    # Check loss patterns
    mnm_losses = [float(row['mnm_loss']) for row in rows[-10:]]
    print(f"Last 10 MNM losses: {mnm_losses}")
    
    if mnm_losses[0] > mnm_losses[-1]:
        print("✅ MNM loss is decreasing (model learning)")
    else:
        print("⚠️ MNM loss not decreasing consistently")

def test_mnm_metric():
    """Test MNM accuracy computation with synthetic data."""
    from src.training.metrics import masked_token_accuracy
    
    # Create synthetic test case
    vocab_size = 100
    seq_len = 10
    batch_size = 1
    
    # Perfect predictions
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[0, :3] = -100  # Mask first 3 positions
    
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    for i in range(seq_len):
        if labels[0, i] != -100:
            logits[0, i, labels[0, i]] = 10.0  # High confidence for correct class
    
    accuracy = masked_token_accuracy(logits, labels)
    print(f"Synthetic test - Expected: 1.0, Got: {accuracy:.4f}")
    
    if abs(accuracy - 1.0) < 0.001:
        print("✅ MNM metric computation is correct")
    else:
        print("🚨 MNM metric has a bug")

if __name__ == "__main__":
    print("=== MNM Diagnostic ===")
    check_recent_training_logs()
    print("\n=== Metric Test ===")
    test_mnm_metric()

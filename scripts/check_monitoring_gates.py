#!/usr/bin/env python3
"""
Monitoring gates script to validate training metrics meet minimum requirements.
Checks for >=10% improvement on val_acc_mnm and non-regression on loss.
"""
from pathlib import Path
import sys
import csv
import argparse

def check_monitoring_gates(metrics_file, improvement_threshold=0.10):
    """
    Check if training metrics meet monitoring gate requirements.
    
    Args:
        metrics_file: Path to training metrics CSV file
        improvement_threshold: Minimum improvement threshold (0.10 = 10%)
    
    Returns:
        tuple: (success: bool, message: str)
    """
    if not Path(metrics_file).exists():
        return False, f"Metrics file does not exist: {metrics_file}"
    
    # Read metrics from CSV
    steps = []
    losses = []
    val_acc_mnms = []
    
    with open(metrics_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            losses.append(float(row['total_loss']))
            val_acc_mnms.append(float(row['mnm_validation_accuracy']))
    
    if len(val_acc_mnms) < 2:
        return False, "Insufficient data points for validation"
    
    # Calculate improvement - compare early and late values
    # For the 10% improvement gate, we'll compare early values to final values
    early_idx = min(10, len(val_acc_mnms) // 4)  # Use value from early in training
    if early_idx >= len(val_acc_mnms):
        early_idx = 0
    
    early_val_acc_mnm = val_acc_mnms[early_idx]
    final_val_acc_mnm = val_acc_mnms[-1]
    
    if early_val_acc_mnm == 0:
        if final_val_acc_mnm == 0:
            improvement = 0
        else:
            improvement = float('inf')  # Improvement from 0 to non-zero is infinite
    else:
        improvement = (final_val_acc_mnm - early_val_acc_mnm) / abs(early_val_acc_mnm)
    
    improvement_percentage = improvement * 100
    
    # Check MNM improvement gate (>= 10%)
    mnm_gate_passed = improvement >= improvement_threshold
    mnm_gate_message = f"MNM validation accuracy improvement: {improvement_percentage:.2f}% (threshold: {improvement_threshold*100:.1f}%) - {'PASS' if mnm_gate_passed else 'FAIL'}"
    
    # Check non-regression gate on loss (final loss should be less than early loss)
    early_loss = losses[early_idx]
    final_loss = losses[-1]
    
    loss_regression = final_loss > early_loss
    loss_gate_passed = not loss_regression
    loss_gate_message = f"Loss regression check: Early loss={early_loss:.4f}, Final loss={final_loss:.4f} - {'PASS' if loss_gate_passed else 'FAIL (LOSS REGRESSION)'}"
    
    # Overall result
    overall_passed = mnm_gate_passed and loss_gate_passed
    overall_message = f"Overall monitoring gates: {'PASS' if overall_passed else 'FAIL'}"
    
    print(f"Early val_acc_mnm: {early_val_acc_mnm:.4f}")
    print(f"Final val_acc_mnm: {final_val_acc_mnm:.4f}")
    print(f"Improvement: {improvement_percentage:.2f}%")
    print(mnm_gate_message)
    print(loss_gate_message)
    print(overall_message)
    
    return overall_passed, f"MNM Improvement: {improvement_percentage:.2f}%, Loss check: {'No regression' if loss_gate_passed else 'Regression detected'}"

def main():
    parser = argparse.ArgumentParser(description="Check training monitoring gates")
    parser.add_argument("--metrics_file", type=str, default="logs/train_metrics.csv", 
                        help="Path to training metrics CSV file")
    parser.add_argument("--improvement_threshold", type=float, default=0.10,
                        help="Minimum improvement threshold (default: 0.10 for 10%)")
    
    args = parser.parse_args()
    
    success, message = check_monitoring_gates(args.metrics_file, args.improvement_threshold)
    
    if success:
        print(f"\n✓ Monitoring gates PASSED: {message}")
        sys.exit(0)
    else:
        print(f"\n✗ Monitoring gates FAILED: {message}")
        sys.exit(1)

if __name__ == "__main__":
    main()
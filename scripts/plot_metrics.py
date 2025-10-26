#!/usr/bin/env python3
"""Generate metrics plots for production releases."""

import csv
import sys
from pathlib import Path

def plot_training_metrics(metrics_file, output_dir="plots"):
    """Generate training metrics plots."""
    
    if not Path(metrics_file).exists():
        print(f"⚠️ Metrics file not found: {metrics_file}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("⚠️ matplotlib not available, skipping plots")
        return False
    
    # Read metrics
    steps, total_loss, mlm_loss, mnm_loss = [], [], [], []
    mlm_acc, mnm_acc = [], []
    
    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            total_loss.append(float(row['total_loss']))
            mlm_loss.append(float(row['mlm_loss']))
            mnm_loss.append(float(row['mnm_loss']))
            
            # Handle different column names
            mlm_col = 'mlm_validation_accuracy' if 'mlm_validation_accuracy' in row else 'mlm_acc'
            mnm_col = 'mnm_validation_accuracy' if 'mnm_validation_accuracy' in row else 'mnm_acc'
            
            mlm_acc.append(float(row.get(mlm_col, 0)))
            mnm_acc.append(float(row.get(mnm_col, 0)))
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(steps, total_loss, label='Total Loss', linewidth=2)
    plt.plot(steps, mlm_loss, label='MLM Loss', alpha=0.7)
    plt.plot(steps, mnm_loss, label='MNM Loss', alpha=0.7)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/training_losses.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(steps, [acc * 100 for acc in mlm_acc], label='MLM Accuracy', linewidth=2)
    plt.plot(steps, [acc * 100 for acc in mnm_acc], label='MNM Accuracy', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.savefig(f"{output_dir}/training_accuracies.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated plots in {output_dir}/")
    print(f"  - training_losses.png")
    print(f"  - training_accuracies.png")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_file", help="Path to metrics CSV file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()
    
    success = plot_training_metrics(args.metrics_file, args.output_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

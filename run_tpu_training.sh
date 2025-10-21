#!/bin/bash

# TPU Training Workflow

# Exit on error
set -e

# 1. Pre-flight checks
echo "Checking for torch_xla..."
if ! python -c "import torch_xla" &> /dev/null; then
    echo "Error: torch_xla not found. Please install with: pip install .[tpu]"
    exit 1
fi
echo "torch_xla found."

# 2. Run TPU training
echo "Starting TPU training..."
python scripts/train.py --config configs/train_tpu.yaml --steps 300 --seed 42
echo "TPU training finished."

# 3. Run monitoring gates
echo "Running monitoring gates..."
python scripts/check_monitoring_gates.py --metrics_file logs/train_metrics.csv
echo "Monitoring gates check finished."

_
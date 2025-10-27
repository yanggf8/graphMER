#!/bin/bash
# Start 10,000-step CPU training run
# Expected time: ~12-13 hours (overnight run)

echo "============================================================"
echo "Starting 10,000-Step Training Run on CPU"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Device: CPU (12th Gen Intel i5-1240P)"
echo "  Steps: 10,000"
echo "  Expected time: ~12.6 hours"
echo "  Checkpoints: Every 1,000 steps"
echo "  Config: configs/train_cpu_optimized.yaml"
echo ""
echo "Start time: $(date)"
echo ""
echo "============================================================"
echo ""

# Create logs directory
mkdir -p logs/checkpoints

# Start training with monitoring
python3 scripts/train_v2.py \
  --config configs/train_cpu_optimized.yaml \
  --steps 10000 \
  --seed 42 \
  --micro_batch_size 1 \
  --grad_accum_steps 8 \
  --save_every_steps 1000 \
  --clip_grad 1.0 \
  --warmup_steps 500 \
  2>&1 | tee logs/train_10k_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "End time: $(date)"
echo ""
echo "Checkpoints saved in: logs/checkpoints/"
echo "Metrics saved in: logs/train_v2_metrics.csv"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: python scripts/eval_comprehensive.py"
echo "  2. Check metrics: cat logs/train_v2_metrics.csv"
echo "  3. Compare to baseline (3.5k steps)"
echo ""

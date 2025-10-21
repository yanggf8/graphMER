#!/bin/bash
# GraphMER-SE Production Verification Script

echo "=== GraphMER-SE Production Verification ==="
echo

# Check 1: Ablation improvement (>= 10% threshold)
echo "1. Verifying MNM improvement >= 10%..."
IMPROVEMENT=$(python scripts/summarize_logs.py --steps 100 2>/dev/null | grep "val_acc_mnm" | grep -o "[0-9.]*%" | head -1 | tr -d '%')
if [[ -z "$IMPROVEMENT" ]]; then
    echo "❌ Could not parse improvement. Run: python scripts/run_ablation.py --config configs/train_cpu.yaml --steps 200"
    exit 1
elif python3 -c "exit(0 if float('$IMPROVEMENT') >= 10 else 1)"; then
    echo "✅ MNM improvement confirmed: ${IMPROVEMENT}%"
else
    echo "❌ MNM improvement ${IMPROVEMENT}% below 10% threshold"
    exit 1
fi

# Check 2: Dataset scale
echo "2. Verifying 30,826 triples and 99.1% validation..."
if grep -q '"total_triples": 30826' data/kg/manifest.json && grep -q '"domain_range_ratio": 0.99' data/kg/manifest.json; then
    echo "✅ Dataset scale confirmed"
else
    echo "❌ Dataset scale verification failed"
    exit 1
fi

# Check 3: Training metrics
echo "3. Verifying 300 samples and 878 vocab..."
if grep -q "300 samples" logs/training_dataset_validation.log && grep -q "Vocab size: 878" logs/training_dataset_validation.log; then
    echo "✅ Training dataset metrics confirmed"
else
    echo "❌ Training dataset metrics verification failed"
    exit 1
fi

echo
echo "🎉 All production claims verified successfully!"

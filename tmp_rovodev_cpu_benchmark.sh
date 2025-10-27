#!/bin/bash
# CPU Training Speed Benchmark
# Runs actual training for 20 steps to measure realistic performance

echo "============================================================"
echo "CPU Training Speed Benchmark"
echo "============================================================"
echo ""
echo "System Information:"
echo "-------------------"
lscpu | grep -E "Model name|CPU\(s\):|Thread|Core|MHz" | head -6
echo ""
free -h | grep -E "Mem:|Swap:"
echo ""
echo "PyTorch Configuration:"
echo "----------------------"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Threads: {torch.get_num_threads()}')"
echo ""
echo "============================================================"
echo "Running 20-step training benchmark..."
echo "============================================================"
echo ""

# Run training for 20 steps and time it
START_TIME=$(date +%s)

python3 scripts/train_v2.py \
    --config configs/train_cpu.yaml \
    --steps 20 \
    --seed 42 \
    --micro_batch_size 1 \
    --grad_accum_steps 4 \
    --save_every_steps 0 \
    2>&1 | tee tmp_rovodev_benchmark_output.log

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
echo ""
echo "Total time for 20 steps: ${ELAPSED} seconds"
echo ""

# Calculate time per step
TIME_PER_STEP=$(echo "scale=2; $ELAPSED / 20" | bc)
echo "Time per effective step: ${TIME_PER_STEP} seconds"
echo "(with grad_accum_steps=4, each effective step = 4 micro-batches)"
echo ""

# Extract actual step times from log if available
echo "Extracting step times from log..."
grep "Step.*loss=" tmp_rovodev_benchmark_output.log | tail -5 || echo "No step logs found"
echo ""

# Calculate projections
echo "============================================================"
echo "Training Time Projections"
echo "============================================================"
echo ""

calculate_time() {
    local steps=$1
    local seconds=$(echo "scale=0; $steps * $TIME_PER_STEP" | bc)
    local minutes=$(echo "scale=1; $seconds / 60" | bc)
    local hours=$(echo "scale=1; $seconds / 3600" | bc)
    local days=$(echo "scale=2; $seconds / 86400" | bc)
    
    if (( $(echo "$hours < 1" | bc -l) )); then
        echo "  $steps steps: ${minutes} minutes"
    elif (( $(echo "$hours < 24" | bc -l) )); then
        echo "  $steps steps: ${hours} hours"
    else
        echo "  $steps steps: ${days} days (${hours} hours)"
    fi
}

calculate_time 100
calculate_time 500
calculate_time 1000
calculate_time 3500
calculate_time 6500
calculate_time 10000

echo ""
echo "Note: These are estimates based on 20 steps. Actual times may vary."
echo "============================================================"

# CPU Training Profile & Benchmark Results

**Date:** October 27, 2025  
**Machine:** 12th Gen Intel(R) Core(TM) i5-1240P (8 cores, 16 threads)  
**Memory:** 19 GB RAM  
**PyTorch:** 2.8.0 (CPU-only, no CUDA)

---

## Benchmark Results Summary

### Test Configuration
- **Steps:** 20 training steps
- **Batch size:** 1 (micro-batch)
- **Gradient accumulation:** 4 steps
- **Effective batch size:** 4
- **Sequence length:** 512 tokens (from train_cpu.yaml)
- **Model:** 768 hidden, 12 layers, 12 heads (80M parameters)

### Measured Performance

| Metric | Value |
|--------|-------|
| **Total time (20 steps)** | 91 seconds |
| **Time per effective step** | 4.55 seconds |
| **Throughput** | 0.22 steps/second |
| **Initialization overhead** | ~60 seconds (dataset building, model loading) |

### Training Time Projections

| Target Steps | Estimated Time | Notes |
|-------------|----------------|-------|
| **100 steps** | 7.5 minutes | Quick test run |
| **500 steps** | 38 minutes | Minimal validation checkpoint |
| **1,000 steps** | 1.3 hours | Initial convergence check |
| **3,500 steps** | 4.4 hours | Current checkpoint level |
| **6,500 steps** | 8.2 hours | Remaining to reach 10k from 3.5k |
| **10,000 steps** | **12.6 hours** | **Target for production evaluation** |

---

## Training Time Analysis

### Current Situation
- We have a checkpoint at **3,500 steps** (already trained)
- Need to reach **10,000 steps** for proper evaluation
- **Remaining:** 6,500 steps = **~8.2 hours**

### Realistic Timeline
```
Start fresh 10k training:  ~12.6 hours (overnight run)
Resume from 3.5k → 10k:    ~8.2 hours (overnight run)
```

### Recommended Approach
**Option A: Resume from checkpoint (RECOMMENDED)**
- Continue from existing 3,500-step checkpoint
- Train additional 6,500 steps
- Total time: ~8-9 hours
- Advantage: Builds on existing progress

**Option B: Fresh 10k training**
- Start from scratch with optimized config
- Train full 10,000 steps  
- Total time: ~12-13 hours
- Advantage: Clean reproducible run with seed 42

---

## Optimized CPU Configuration

### Key Optimizations for This Hardware

```yaml
# configs/train_cpu_optimized.yaml

run:
  seed: 42
  mixed_precision: false  # CPU doesn't benefit from fp16
  
hardware:
  device: cpu
  num_workers: 4  # Good for 8-core CPU
  
training_data:
  micro_batch_size: 1      # Small batch to fit in CPU RAM
  grad_accumulation_steps: 8  # Effective batch = 8
  max_seq_len: 128         # Reduced from 512 for speed
  
model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  use_rel_attention_bias: true
  
optimizer:
  lr: 3.0e-4
  weight_decay: 0.01
  
checkpointing:
  save_every_steps: 1000   # Save checkpoints every 1k steps
  activation_checkpointing: true
  gradient_checkpointing: true
```

### Performance Tuning Options

**Speed up training (sacrifice some quality):**
```yaml
training_data:
  max_seq_len: 128         # 4x faster than 512
  micro_batch_size: 2      # If RAM permits
  
model:
  num_layers: 8            # 33% faster (8 vs 12 layers)
  hidden_size: 512         # Smaller model
```

**Projected speedup:** 4-6x faster → 10k steps in ~2-3 hours

---

## CPU Performance Bottlenecks

### Identified Issues
1. **No hardware acceleration** - CPU lacks tensor cores
2. **Sequential processing** - No parallel GPU streams
3. **Memory bandwidth** - System RAM slower than GPU VRAM
4. **Attention computation** - O(n²) very expensive on CPU

### What Works Well
✅ **Dataset building** - Fast, only ~10 seconds  
✅ **Tokenization** - Efficient BPE implementation  
✅ **Model convergence** - Loss decreasing properly  
✅ **Memory usage** - Fits comfortably in 19GB RAM  

### What's Slow
❌ **Forward pass** - Transformer attention on CPU  
❌ **Backward pass** - Gradient computation  
❌ **Large sequence lengths** - 512 tokens = 262k attention ops  

---

## Recommendations

### For Immediate Training (Next 24 Hours)

**RECOMMENDED: Overnight 10k Run**
```bash
# Start tonight, complete by morning
python scripts/train_v2.py \
  --config configs/train_cpu.yaml \
  --steps 10000 \
  --seed 42 \
  --micro_batch_size 1 \
  --grad_accum_steps 8 \
  --save_every_steps 1000 \
  --clip_grad 1.0 \
  --warmup_steps 500
```

**Expected outcome:**
- Start: 6:00 PM today
- Complete: 6:30 AM tomorrow (~12.5 hours)
- Checkpoints saved at: 1k, 2k, 3k, 4k, 5k, 6k, 7k, 8k, 9k, 10k steps
- Can evaluate incrementally tomorrow

### For Faster Iteration (Testing)

**Quick 1k steps for validation (~1.3 hours)**
```bash
# Verify config and convergence
python scripts/train_v2.py \
  --config configs/train_cpu.yaml \
  --steps 1000 \
  --seed 42 \
  --micro_batch_size 1 \
  --grad_accum_steps 4 \
  --save_every_steps 500
```

### For Production (When GPU Available)

**Same config on GPU: ~10-20 minutes for 10k steps**
- 30-60x faster than CPU
- Can iterate rapidly
- Recommended for hyperparameter tuning

---

## Next Steps

### Immediate (Today)
1. ✅ CPU benchmark complete
2. ⏭️ **Start 10k training run overnight**
3. ⏭️ Monitor first 100 steps for stability
4. ⏭️ Check checkpoint at 1k steps tomorrow morning

### Tomorrow (After 10k Completion)
1. Run comprehensive evaluation on 10k checkpoint
2. Compare metrics to 3.5k baseline
3. Verify link prediction MRR improvement (target: >0.3)
4. Decide: continue to 15k or scale up data?

### Future (When Available)
1. Migrate to GPU for faster iteration
2. Scale to 30k KG triples
3. Train to 20k+ steps on full dataset
4. Production deployment

---

## Cost-Benefit Analysis

### CPU Training
- **Cost:** Free (using available hardware)
- **Time:** 12.6 hours for 10k steps
- **Feasibility:** ✅ Overnight run is practical
- **Risk:** Low (no cost if results are poor)

### GPU Alternative (Cloud)
- **Cost:** $5-20 for 1 hour GPU time
- **Time:** 15-30 minutes for 10k steps
- **Feasibility:** ✅ If budget available
- **Risk:** Low cost, much faster iteration

### Recommendation
**Use CPU for this 10k run** because:
1. Overnight training is acceptable timeline
2. No incremental cost
3. Validates that extended training helps
4. Can move to GPU after confirming approach

If 10k checkpoint shows improvement (MRR >0.3), then GPU becomes worthwhile for scaling to 20k+ steps.

---

## Monitoring Commands

### Check training progress:
```bash
# View latest metrics
tail -20 logs/train_v2_metrics.csv

# Monitor in real-time
watch -n 10 'tail -5 logs/train_v2_metrics.csv'

# Check system resources
htop  # CPU usage
nvidia-smi  # If GPU ever available
```

### Evaluate checkpoint mid-training:
```bash
# Quick eval at 5k steps
python scripts/eval_comprehensive.py \
  --checkpoint logs/checkpoints/model_v2_step5000_*.pt \
  --quick
```

---

## Summary

✅ **CPU is viable for 10k training** (12.6 hours)  
✅ **Performance is predictable** (4.55 sec/step)  
✅ **Model is learning** (loss decreasing in benchmark)  
✅ **Ready to start extended training**  

**Action:** Start 10k training overnight for results by morning.


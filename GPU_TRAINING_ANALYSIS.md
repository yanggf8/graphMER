# GraphMER-SE: RTX 3050 vs TPU Training Analysis

**Date:** October 20, 2025  
**Question:** Is RTX 3050 better than Colab TPU for GraphMER-SE training?

---

## Hardware Comparison

### RTX 3050 Specs
- **VRAM:** 8 GB GDDR6
- **CUDA Cores:** 2,560
- **Tensor Cores:** 80 (3rd gen)
- **Memory Bandwidth:** 224 GB/s
- **TDP:** 130W
- **FP16 Performance:** ~9 TFLOPS
- **Cost:** Already owned (no hourly charges)
- **Availability:** 24/7 unlimited

### Google Colab TPU v2-8 (Free Tier)
- **Memory:** 64 GB HBM (8 GB per core)
- **Cores:** 8
- **Memory Bandwidth:** 600 GB/s
- **BF16 Performance:** ~180 TFLOPS
- **Cost:** Free
- **Limitations:**
  - 12-hour session limit
  - 90-minute idle timeout
  - Weekly quota (~20-30 hours)
  - Session can disconnect unexpectedly

---

## GraphMER-SE Memory Requirements

### Model Size Analysis

**Current Configuration:**
```yaml
model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  intermediate_size: 3072
  dropout: 0.1
```

**Estimated Model Size:**
- Parameters: ~80M (target from specs)
- Model weights (FP32): ~320 MB
- Model weights (FP16): ~160 MB
- Optimizer state (AdamW): ~640 MB (2x params)
- Gradients: ~160 MB
- **Total (training):** ~1.1 GB

### Batch Memory Requirements

**Per-sample Memory (sequence_len=768):**
- Input embeddings: ~3 MB
- Attention matrices: ~36 MB (768x768x12 layers)
- Activations: ~50 MB
- **Total per sample:** ~90 MB

**Batch Size Calculations (RTX 3050 with 8GB):**
- Available for batches: 8 GB - 1.1 GB (model) = 6.9 GB
- Theoretical max batch: 6900 MB / 90 MB ≈ **76 samples**
- Practical batch (with overhead): **32-48 samples**

### Dataset Size
- **Training samples:** 300 (from validation report)
- **Knowledge graph:** 30,826 triples (~7 MB)
- **Fits entirely in memory:** ✅ Yes

---

## Training Performance Comparison

### RTX 3050 (Estimated)
**Configuration:**
```yaml
hardware:
  device: cuda
  batch_size: 32  # Can go higher if needed
  mixed_precision: fp16  # Use automatic mixed precision
  num_workers: 4

training:
  gradient_accumulation_steps: 1  # No need, batch fits
```

**Expected Performance:**
- Steps per second: ~5-8 steps/sec (with FP16)
- Time for 1000 steps: ~2-3 minutes
- Time for 10,000 steps: ~20-30 minutes
- Time for 50,000 steps: ~2-3 hours

**Advantages:**
- ✅ No session limits (train for days if needed)
- ✅ No weekly quotas
- ✅ No disconnections
- ✅ Faster iteration (local access)
- ✅ Can use full batch size (32-48)
- ✅ Better for debugging (immediate access)

### Colab TPU v2-8 (Current)
**Configuration:**
```yaml
hardware:
  device: tpu
  tpu_cores: 8
  batch_size: 2  # Per core = 16 total
  mixed_precision: bf16
  gradient_accumulation_steps: 16  # Effective batch = 256
```

**Expected Performance:**
- Steps per second: ~1-2 steps/sec (after XLA compilation)
- Time for 1000 steps: ~15-20 minutes
- Time for 10,000 steps: ~2-3 hours
- Time for 50,000 steps: ~10-15 hours (multiple sessions)

**Limitations:**
- ❌ 12-hour session limit (need to resume)
- ❌ Weekly quota limits
- ❌ Can disconnect randomly
- ❌ Slower iteration (network latency)
- ❌ Small per-core batch (2)
- ❌ XLA compilation overhead

---

## Recommendation

### ✅ **Use RTX 3050 for GraphMER-SE Training**

**Reasons:**

1. **Model Fits Comfortably:**
   - 80M params + optimizer state = ~1.1 GB
   - 8 GB VRAM is more than sufficient
   - Can use batch size 32-48 (vs TPU's 16)

2. **Better Training Experience:**
   - No session limits or disconnections
   - Faster iteration for debugging
   - Full control over training schedule
   - Can train for days continuously

3. **Comparable or Better Speed:**
   - RTX 3050: ~5-8 steps/sec with FP16
   - TPU: ~1-2 steps/sec (with compilation overhead)
   - RTX 3050 likely **2-4x faster** in practice

4. **Cost Effective:**
   - Already owned hardware
   - No cloud costs
   - No quota limitations

5. **Development Friendly:**
   - Local access for debugging
   - Can profile with CUDA tools
   - Easier checkpoint management
   - No Drive upload/download needed

---

## GPU Training Configuration

### Create GPU Config

```yaml
# configs/train_gpu.yaml
run:
  seed: 1337
  epochs: 5
  gradient_accumulation_steps: 1  # Not needed with larger batch
  log_interval: 50
  eval_interval_steps: 500
  save_interval_steps: 1000
  mixed_precision: fp16  # Use automatic mixed precision
  deterministic: true

hardware:
  device: cuda
  num_workers: 4  # Adjust based on CPU cores

model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  intermediate_size: 3072
  dropout: 0.1
  positional_encoding: alibi
  norm: rmsnorm
  activation: swiglu
  hgat:
    enabled: true
    relation_bias: true
  use_rel_attention_bias: true

optimizer:
  name: adamw
  lr: 3.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.98]
  eps: 1e-8
  scheduler:
    name: cosine
    warmup_steps: 1500

training_data:
  max_seq_len: 768
  micro_batch_size: 32  # Larger than TPU (2 per core)
  pack_sequences: true
  short_to_long_curriculum:
    enabled: true
    schedule:
      - {steps: 0, max_seq_len: 512}
      - {steps: 15000, max_seq_len: 768}

objectives:
  mlm:
    mask_prob: 0.15
    span_mask_identifiers: true
  mnm:
    mask_prob: 0.20
    type_consistent_negatives: 2
    hard_negatives: 2

encoding:
  leaves_per_anchor:
    positive: 2
    negatives: 2
  max_leaves_per_sequence: 10

regularizers:
  ontology_constraints:
    antisymmetry_weight: 0.2
    acyclicity_weight: 0.2
  contrastive:
    enabled: true
    temperature: 0.07

checkpointing:
  activation_checkpointing: false  # Not needed with 8GB VRAM
```

---

## Setup Instructions for RTX 3050

### 1. Install CUDA Dependencies

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. Install GraphMER Dependencies

```bash
cd /path/to/graphMER
pip install -e .
```

### 3. Run Training

```bash
# Quick test (100 steps, ~30 seconds)
python scripts/train.py --config configs/train_gpu.yaml --steps 100

# Full training (10,000 steps, ~30 minutes)
python scripts/train.py --config configs/train_gpu.yaml --steps 10000

# Production run (50,000 steps, ~2-3 hours)
python scripts/train.py --config configs/train_gpu.yaml
```

### 4. Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Or in Python
python -c "import torch; torch.cuda.memory_summary()"
```

---

## Memory Optimization Tips (if needed)

If you encounter OOM (Out of Memory) errors:

### 1. Reduce Batch Size
```yaml
training_data:
  micro_batch_size: 16  # Reduce from 32
```

### 2. Enable Gradient Checkpointing
```yaml
checkpointing:
  activation_checkpointing: true  # Trades compute for memory
```

### 3. Reduce Sequence Length
```yaml
training_data:
  max_seq_len: 512  # Reduce from 768
```

### 4. Use Gradient Accumulation
```yaml
run:
  gradient_accumulation_steps: 2  # Effective batch = 16 * 2 = 32
training_data:
  micro_batch_size: 16
```

---

## Performance Benchmarking

### Expected Training Times (RTX 3050)

| Steps | Estimated Time | Notes |
|-------|----------------|-------|
| 100 | ~30 seconds | Quick smoke test |
| 1,000 | ~2-3 minutes | Initial validation |
| 10,000 | ~20-30 minutes | Short production run |
| 50,000 | ~2-3 hours | Full training run |
| 100,000 | ~4-6 hours | Extended training |

**vs Colab TPU:**
- TPU: ~15-20 min for 1000 steps
- RTX 3050: ~2-3 min for 1000 steps
- **Speed advantage: 5-10x faster** (no XLA compilation overhead)

---

## When to Use Each Platform

### Use RTX 3050 ✅ (Recommended)
- ✅ Main training runs
- ✅ Development and debugging
- ✅ Hyperparameter tuning
- ✅ Quick iterations
- ✅ Long training runs (>12 hours)
- ✅ Anytime you need reliability

### Use Colab TPU (Optional)
- ✅ When RTX 3050 is unavailable
- ✅ For very large models (>500M params)
- ✅ To validate TPU compatibility
- ✅ As backup infrastructure
- ✅ For distributed training experiments

---

## Conclusion

**Verdict: RTX 3050 is BETTER for GraphMER-SE training**

**Key Advantages:**
1. ✅ Model fits comfortably (1.1 GB < 8 GB)
2. ✅ 2-10x faster than Colab TPU
3. ✅ No session limits or quotas
4. ✅ Better development experience
5. ✅ More reliable (no disconnections)

**Action Items:**
1. Create `configs/train_gpu.yaml` (see above)
2. Install CUDA PyTorch on RTX 3050 machine
3. Copy project to RTX 3050 machine
4. Run training locally

**Use Colab TPU only as backup or for very long runs when RTX 3050 is busy.**

---

**Recommendation:** ✅ **Train on RTX 3050 for best results**

**Created:** 2025-10-20  
**Analysis:** RTX 3050 (8GB) vs Colab TPU v2-8  
**Model:** GraphMER-SE (~80M params)

# RTX 3070 Training Guide

## Hardware Specifications
- **GPU:** NVIDIA GeForce RTX 3070
- **VRAM:** 8GB GDDR6
- **Compute Capability:** 8.6
- **CPU Cores:** 6
- **System RAM:** 15GB

## Profile Configuration

### Profile Name: `RTX3070_8G`

**Optimized Parameters:**
```yaml
steps: 4000
max_samples: 20000
micro_batch_size: 8
grad_accum_steps: 16
save_every_steps: 250
amp: true
config: "configs/train_v2_gpu.yaml"
use_full_kg: true
warmup_steps: 400
clip_grad: 1.0
mnm_weight_ramp: 1000
max_code_files: 150
```

**Effective Batch Size:** 8 × 16 = 128 (optimized for RTX 3070 throughput)

## Usage

### Basic Training
```bash
python3 scripts/run_gpu_profile.py --profile RTX3070_8G
```

### Extended Training (5000 steps)
```bash
python3 scripts/run_gpu_profile.py --profile RTX3070_8G --steps 5000
```

### Custom Overrides
```bash
python3 scripts/run_gpu_profile.py --profile RTX3070_8G \
  --micro_batch_size 10 \
  --save_every_steps 500 \
  --steps 6000
```

### Full Training Run
```bash
python3 scripts/run_gpu_profile.py --profile RTX3070_8G \
  --steps 10000 \
  --max_samples 30000
```

## Performance Expectations

**Comparison to Other Hardware:**
- **RTX 3050 (8GB):** RTX 3070 is ~30% faster (higher CUDA cores, memory bandwidth)
- **RTX 4060 Ti (16GB):** Similar performance, but 3070 limited to 8GB VRAM
- **Apple M2 (MPS):** RTX 3070 is ~50-70% faster with AMP enabled
- **CPU (i5-1240P):** RTX 3070 is ~60-80x faster

**Estimated Training Speed:**
- **Steps/sec:** ~1.5-2.0 (with AMP)
- **4000 steps:** ~35-45 minutes
- **10000 steps:** ~1.5-2 hours

## Optimization Tips

### 1. Maximize Throughput
Increase `micro_batch_size` if VRAM allows:
```bash
python3 scripts/run_gpu_profile.py --profile RTX3070_8G \
  --micro_batch_size 10
```

### 2. Memory Optimization
If encountering OOM errors, reduce batch size:
```bash
python3 scripts/run_gpu_profile.py --profile RTX3070_8G \
  --micro_batch_size 6 \
  --grad_accum_steps 22
```

### 3. Monitor GPU Utilization
```bash
# Terminal 1: Training
python3 scripts/run_gpu_profile.py --profile RTX3070_8G

# Terminal 2: Monitoring
watch -n 1 nvidia-smi
```

### 4. Curriculum Learning
The base config `train_v2_gpu.yaml` includes progressive sequence length ramping:
- Steps 0-1000: 256 tokens
- Steps 1000-2500: 384 tokens
- Steps 2500-4000: 512 tokens
- Steps 4000+: 768 tokens

## Key Features

✅ **Full KG Support:** Uses all 29k knowledge graph triples
✅ **Mixed Precision (AMP):** FP16 for 2x speed boost
✅ **Gradient Clipping:** Prevents training instability
✅ **Warmup Schedule:** 400-step linear warmup for stable convergence
✅ **MNM Weight Ramping:** Gradually increases MNM loss weight over 1000 steps
✅ **Extended Code Files:** Processes up to 150 raw code files for richer training data

## Troubleshooting

### OOM Error
```bash
# Reduce batch size
python3 scripts/run_gpu_profile.py --profile RTX3070_8G \
  --micro_batch_size 6
```

### Slow Performance
```bash
# Verify AMP is enabled
grep "amp: true" configs/gpu_profiles.yaml

# Check GPU utilization
nvidia-smi
```

### Loss Divergence
```bash
# Increase gradient clipping
python3 scripts/run_gpu_profile.py --profile RTX3070_8G \
  --clip_grad 0.5
```

## Expected Results

Based on validated profiles (408032G, M2_8C_16G), you should expect:
- **Total Loss Reduction:** 45-55% over 4000 steps
- **MLM Accuracy:** 35-40%
- **MNM Loss:** Stable convergence below 2.0
- **Checkpoints:** Saved every 250 steps in `outputs/checkpoints/`

## Environment Setup

### Prerequisites Validation

Run the following to verify your environment is ready:

```bash
# Check GPU detection
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Expected output:
# CUDA available: True
# GPU: NVIDIA GeForce RTX 3070
```

### Initial Setup (One-time)

If this is your first time, you need to install dependencies and build the knowledge graph:

```bash
# 1. Install dependencies (if not already done)
~/.local/bin/pip install --user -r requirements.txt

# 2. Build knowledge graph (generates data/kg/seed_multilang.jsonl)
python3 scripts/build_kg.py

# 3. Verify with a quick test run
python3 scripts/run_gpu_profile.py --profile RTX3070_8G --steps 10 --max_samples 100
```

### Environment Details

**Installed Components:**
- PyTorch 2.9.0+ with CUDA 12.8 support
- Knowledge graph: 28,961 triples from 219 code files
- BPE tokenizer: 8,000 vocab size (auto-trained on first run)

**Output Locations:**
- Checkpoints: `logs/checkpoints/`
- Metrics: `logs/train_v2_metrics.csv`
- KG Data: `data/kg/seed_multilang.jsonl`
- Tokenizer: `data/tokenizer/code_bpe.json`

## Next Steps

After training, evaluate your model:
```bash
python3 scripts/eval_comprehensive.py --checkpoint logs/checkpoints/model_v2_*.pt
```

Compare results with baseline in `docs/EVALUATION_SUMMARY.md`.

# Apple M2 Training Guide (MPS)

## Hardware Snapshot
- **SoC**: Apple M2 (8-core CPU, 10-core GPU)
- **Memory**: 16GB unified (shared CPU/GPU)
- **Acceleration**: Metal Performance Shaders (MPS) backend via PyTorch 2.1+
- **Storage**: NVMe SSD (required for fast KG + tokenizer access)

## Profile Configuration

### Profile Name: `M2_8C_16G`

```yaml
steps: 3600
max_samples: 20000
micro_batch_size: 4
grad_accum_steps: 24
save_every_steps: 300
amp: false
config: "configs/train_mps.yaml"
use_full_kg: true
warmup_steps: 400
clip_grad: 1.0
mnm_weight_ramp: 900
max_code_files: 120
```

- **Effective Batch Size**: 4 × 24 = 96 tokens per optimizer step.
- **Curriculum**: Sequence length ramps 192 → 384 across four stages.
- **Tokenizer**: `data/tokenizer/code_bpe.json` (auto-regenerated, 13.8k vocab).

## Usage

### Calibrated Training Run
```bash
python3 scripts/run_gpu_profile.py --profile M2_8C_16G
```

### Longer Run (5k+ steps)
```bash
python3 scripts/run_gpu_profile.py --profile M2_8C_16G --steps 5200
```

### Debug / Quick Smoke Test
```bash
python3 scripts/run_gpu_profile.py --profile M2_8C_16G --steps 50 --max_samples 400
```

## Observed Performance (October 28, 2025)
- **Runtime**: ~12 minutes for 3,600 steps (with warmup + full KG sampling).
- **Final Losses**: total **3.65**, MLM **2.16**, MNM **1.84**.
- **Validation Peaks**: MLM accuracy 1.0; MNM accuracy 0.60 on smaller batches.
- **Dataset**: 674 Leafy Chain samples from `data/kg/enhanced_multilang.jsonl`.
- **Checkpoint**: `logs/checkpoints/model_v2_20251028_181737_s42.pt`.
- **Metrics File**: `logs/train_v2_metrics.csv` (overwritten each run).

## Optimization Tips
- **Throughput**: Keep Activity Monitor closed while training; it can throttle MPS jobs.
- **Micro Batching**: Increase `--micro_batch_size` to 5 if you downscale `max_seq_len` to 320.
- **Gradient Stability**: Keep `--clip_grad 1.0`; lowering caused divergence above 3000 steps.
- **Tokenizer Health**: If you see enum errors, rerun `python3 src/training/tokenizer_bpe.py` to regenerate the 13.8k-vocab tokenizer.
- **Artifact Hygiene**: Rename logs/checkpoints after long runs to avoid auto-pruning during future sessions.

## Validation & Follow-Up
1. Generate metadata after a run:
   ```bash
   make generate-metadata RUN_NAME=production_$(date +%Y%m%d_%H%M%S)
   make validate-metadata-prod
   ```
2. Evaluate the resulting checkpoint:
   ```bash
   python3 scripts/eval_comprehensive.py \
     --checkpoint logs/checkpoints/model_v2_20251028_181737_s42.pt
   ```
3. For regression tracking, snapshot `logs/train_v2_metrics.csv` to `logs/runs/<run_name>/`.

## Troubleshooting
- **`data did not match any variant` errors**: Regenerate tokenizer (see above).
- **Slow startup**: The full KG path expands 120 code files; run once to prime OS cache.
- **Val accuracy stuck at 0**: Increase `--mnm_weight_ramp` to 1200 or reduce `max_samples`.
- **Timeouts in automation**: Use `timeout_ms=900000` when invoking profiles from scripts.

## Reference Materials
- Profile definition: `configs/gpu_profiles.yaml` (`M2_8C_16G` entry)
- Curriculum + optimizer settings: `configs/train_mps.yaml`
- Contributor workflow: `AGENTS.md`
- High-level status: `README.md` → “Training Results” section

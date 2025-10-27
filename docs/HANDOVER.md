# GraphMER-SE Project Handover

## Current Status: Implementation Complete, Evaluation Baseline Below Targets

Date: 2025-10-28
Model: 85M parameter encoder implemented and trained; evaluation baselines require more data and task generators.

### Latest Evaluation Update
- Checkpoint loading fix applied; evaluation now uses trained weights.
- Tokenizer persisted and enhanced KG built.
- Link Prediction (enhanced KG): MRR 0.0143, Hits@10 2.6% (3,151 test triples).
- Other tasks at 0% due to missing test-case generation.
- Action: scale KG, add task generators (code search, disambiguation, call-graph, dependency), extend training to 20k–50k steps with hard negatives.

### Quick Start Commands (Updated)

**Run Validated Baseline (500 steps)**:
```bash
python scripts/train.py --config configs/train_cpu.yaml --steps 500 --limit 1000 --chunk_size 10
```

**Run Extended Training (1000+ steps, recommended)**:
```bash
python scripts/train.py --config configs/train_cpu.yaml --steps 1000 --limit 32 --chunk_size 2
```

**Build Enhanced Knowledge Graph**:
```bash
python3 scripts/build_kg_enhanced.py --source_dir data/raw/python_samples --max_files 300
# For larger coverage:
python3 scripts/build_kg_enhanced.py --source_dir data/raw/python_corpus --max_files 5000
```

**Comprehensive Evaluation (fixed loader + enhanced KG)**:
```bash
python3 scripts/eval_comprehensive.py \
  --checkpoint logs/checkpoints/model_v2_20251028_013256_s42.pt \
  --triples data/kg/enhanced_multilang.jsonl \
  --output logs/evaluation_results_enhanced.json
```

**Run Scaling Experiment**:
```bash
python scripts/train.py --config configs/train_scaling.yaml
```

## Architecture Overview

**Model**: GraphMER-SE encoder (85M parameters)
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Intermediate size: 3072
- Relation attention bias: Enabled

**Knowledge Graph**: 29,174 triples
- Domain/range validation: 99.39%
- Acyclic inheritance: Validated
- Multi-language support: Python + Java

**Training Objectives**:
- MLM (Masked Language Modeling)
- MNM (Masked Node Modeling) - knowledge graph task

## Free Training Options

1. CPU Training (always available):
  ```bash
  python scripts/train.py --config configs/train_cpu.yaml --steps 500
  ```

2. Local GPU (if available):
  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 scripts/train_v2.py \
    --config configs/train_v2_gpu.yaml \
    --steps 1000 --max_samples 5000 \
    --amp --micro_batch_size 4 --grad_accum_steps 16 \
    --save_every_steps 200
  ```

## Repository Structure

- `src/` - Core implementation
  - `models/encoder.py` - 85M parameter GraphMER-SE model
  - `training/` - Dataset builder, metrics, evaluator
  - `parsing/` - Python/Java parsers
  - `kg/` - Knowledge graph builder
  - `ontology/` - KG validator
- `configs/` - Training configurations
  - `train_cpu.yaml` - CPU training (validated)
  - `train_v2_gpu.yaml` - Local GPU training (FP16)
  - `train_scaling.yaml` - Large-scale production training
  - `train_tpu.yaml` - TPU training
- `scripts/` - Training and evaluation scripts
  - `train.py` - Main training script
  - `eval.py` - Config-aware evaluation (updated)
- `data/kg/` - Generated knowledge graph (29k+ triples)
- `docs/` - Specifications and documentation

## Next Steps

1. **Scale Training**: Run 1000+ steps for production baseline (commands updated)
2. **Scaling Experiments**: Test train_scaling.yaml for long production runs
3. **Platform Testing**: Validate on ModelScope/Colab for longer runs
4. **Evaluation**: Use config-aware eval.py with proper dimensions on larger datasets
5. **Knowledge Graph Expansion**: Scale to 50k+ triples

## Validation Commands

Verify all production claims:
```bash
# Verify training works
python scripts/train.py --config configs/train_cpu.yaml --steps 100

# Verify KG quality
cat data/kg/manifest.json | grep -E "total_triples|domain_range_ratio"

# Run tests
python -m pytest tests/ -v
```

## License
Personal project with permissively licensed components (MIT/Apache-2.0/BSD/MPL-2.0).

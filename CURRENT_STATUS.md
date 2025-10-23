# GraphMER-SE: Current Status

## Overview

GraphMER-SE is a neurosymbolic encoder for software engineering that combines code tokens with knowledge graph triples. This document tracks the current state of the project.

---

## Model Specifications

### Target Architecture
- **Model Size**: ~85M parameters
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Intermediate Size**: 3072
- **Relation Attention Bias**: Enabled ✅

### Validated Features
- ✅ Leafy Chain Graph Encoding
- ✅ Relation-aware attention (HGAT/attention bias)
- ✅ Multi-language support (Python + Java)
- ✅ Curriculum learning (384→512 seq length)
- ✅ MLM + MNM dual objectives

---

## Recent History

### Critical Bug Discovery & Fix (2024)

**Issue**: Training script (`scripts/train.py`) had hardcoded small model dimensions
- Used 3.2M parameter model instead of intended 85M model
- All previous training runs affected

**Resolution**:
- ✅ Fixed training script to read all dimensions from config
- ✅ Added architecture logging for verification
- ✅ Validated 85M model trains successfully
- ✅ Archived previous 3.2M results to `archive/modelscope_3.2M_results/`

**Current State**: Training script now works correctly with all config files

---

## Training Status

### Completed
- ✅ **500-step baseline** with 85M model
  - Loss reduction: 51.2% (0.1872 → 0.0913)
  - MLM peak accuracy: 81.82%
  - MNM peak accuracy: 29.03%
  - Training time: ~7-8 minutes (CPU)
  - Status: Production baseline established

- ✅ **100-step baseline** with 85M model
  - Loss reduction: 32.0%
  - MLM peak accuracy: 81.82%
  - Training time: ~90 seconds
  - Status: First correct baseline established

### In Progress
- 📋 **1000+ step training** with 85M model for comprehensive baseline
- 📋 Full dataset evaluation (29,174 triples available)
- 📋 Scaling experiments with new config architecture

### Archived (3.2M Model Results)
- ⚠️ 500-step training (45.3% loss reduction)
- ⚠️ 1000-step training (50.9% loss reduction)
- ⚠️ All results moved to `archive/modelscope_3.2M_results/`
- Note: Valid for 3.2M model, but not representative of 85M target

---

## Dataset Status

### Available Data
- **Total Triples**: 29,174 (production-ready)
- **Languages**: Python (primary), Java (integrated)
- **Ontology Validation**: 99.39% consistency
- **Quality**: Multi-file, production-scale

### Current Usage
- **Samples Used**: 100 (in 85M baseline)
- **Vocabulary**: 339 tokens
- **Utilization**: 0.34% of available data

### Scaling Potential
- ✅ 5000+ samples tested (with 3.2M model)
- ✅ Infrastructure supports full dataset
- 📋 Need to test with 85M model at scale

---

## Configuration Files

All configs now use correct baseline dimensions (768/12/12/3072):

### Primary Configs
1. **`configs/train_cpu.yaml`** ✅
   - Target: CPU training
   - Batch size: 1 (gradient accumulation: 64)
   - Learning rate: 2.5e-4
   - Notes: Slow but viable, use high gradient accumulation

2. **`configs/train_gpu.yaml`** ✅
   - Target: NVIDIA RTX 3050 (8GB VRAM)
   - Batch size: 32
   - Learning rate: 3.0e-4
   - Mixed precision: FP16
   - Expected: 5-8 steps/sec

3. **`configs/train_scaling.yaml`** ✅ NEW
   - Target: Large-scale production training
   - Architecture: Same baseline 768/12/12/3072
   - Batch size: 4 (micro), 32 gradient accumulation
   - Mixed precision: BF16
   - Max steps: 50,000 with curriculum learning
   - Optimized for: Extended training runs, evaluation benchmarks

4. **`configs/train_tpu.yaml`**
   - Target: Colab TPU v2
   - Batch size: 128 per core
   - Mixed precision: BF16

5. **`configs/train_kaggle.yaml`**
   - Target: Kaggle GPU environment
   - Optimized for platform constraints

### Deprecated
- `configs/modelscope_config.yaml` - Keep for reference but use standard configs

---

## Infrastructure Status

### Training Pipeline
- ✅ Training script fixed (`scripts/train.py`)
- ✅ Model architecture logging added
- ✅ All dimensions read from config
- ✅ Curriculum learning implemented
- ✅ Validation framework in place

### Evaluation Framework
- ✅ MRR (Mean Reciprocal Rank) metrics
- ✅ Hits@k evaluation
- ✅ Per-task accuracy tracking
- ✅ Loss decomposition (MLM + MNM)
- ✅ Config-aware evaluation (reads model dimensions from config)
- ✅ Checkpoint compatibility (proper state_dict key handling)

### Data Pipeline
- ✅ KG builder (Python + Java parsers)
- ✅ Leafy Chain Packer
- ✅ Manifest-based reproducibility
- ✅ Ontology validation (99.39%)

### Documentation
- ✅ Bug report archived
- ✅ Training guides updated
- ✅ README cleaned up
- ✅ Configuration docs current

---

## Performance Benchmarks

### 85M Model (100 Steps)
```
Model: 768/12/12/3072 (~85M params)
Dataset: 100 samples, 339 vocab

Loss Reduction: 32.0%
MLM Peak Accuracy: 81.82%
MNM Peak Accuracy: 21.43%
Training Time: ~90 seconds
Device: CPU
```

### 3.2M Model (Archived)
```
Model: 256/4/4/1024 (~3.2M params)
Dataset: Various (100-1000 samples)

Best Loss Reduction: 50.9% (1000 steps)
Best MLM Peak: 81.82%
Best MNM Peak: 41.03%
Note: Results archived, not representative of target model
```

---

## Next Steps

### Immediate (High Priority)

#### 1. Establish 500-Step Baseline ⭐
```bash
python scripts/train.py \
  --config configs/train_cpu.yaml \
  --steps 500 \
  --limit 1000 \
  --chunk_size 10
```
**Goal**: Proper baseline for 85M model convergence  
**Expected**: MNM task should improve significantly  
**Time**: ~7-8 minutes on CPU

#### 2. Scale Dataset
```bash
python scripts/train.py \
  --config configs/train_cpu.yaml \
  --steps 500 \
  --limit 5000 \
  --chunk_size 50
```
**Goal**: Test generalization with larger dataset  
**Expected**: Larger vocab, better diversity  
**Time**: ~10-12 minutes

#### 3. GPU Training Test
```bash
python scripts/train.py \
  --config configs/train_gpu.yaml \
  --steps 500 \
  --limit 1000 \
  --chunk_size 10
```
**Goal**: Validate GPU acceleration  
**Expected**: 5-10x faster than CPU  
**Time**: ~1-2 minutes on RTX 3050

### Medium Term

4. **Comprehensive Evaluation**
   - Run full evaluation suite on 500-step model
   - Compare CPU vs GPU training quality
   - Document baseline metrics

5. **Hyperparameter Tuning**
   - Learning rate sweep [1e-4, 2.5e-4, 5e-4]
   - Batch size experiments
   - Curriculum schedule optimization

6. **Platform Migration**
   - Deploy to Kaggle GPU
   - Set up Colab TPU training
   - Cross-platform benchmarking

### Long Term

7. **Full Dataset Training**
   - Scale to 10,000+ samples from 29,174 triples
   - Multi-language integration (full Java dataset)
   - Production-scale validation

8. **Model Optimization**
   - Test larger models (512 hidden → 1024)
   - Experiment with architecture variants
   - Validate mixed precision training

9. **Production Deployment**
   - Automated training pipelines
   - Model registry and versioning
   - Monitoring and alerting

---

## Key Metrics to Track

### Training Metrics
- Total loss reduction (target: >40%)
- MLM loss and accuracy
- MNM loss and accuracy
- Training steps per second
- Memory usage

### Evaluation Metrics
- MLM MRR and Hits@10
- MNM MRR and Hits@10
- Per-relation accuracy
- Generalization to unseen code

### Infrastructure Metrics
- Training time per step
- GPU/TPU utilization
- Dataset loading time
- Checkpoint size

---

## Risk Assessment

### Low Risk ✅
- Training pipeline stability
- Configuration management
- Dataset quality
- Bug prevention (logging added)

### Medium Risk ⚠️
- CPU training speed (viable but slow)
- MNM task convergence (needs more steps)
- Memory constraints at scale

### Mitigation
- Use GPU/TPU for faster iteration
- Increase training steps for MNM task
- Gradient checkpointing for memory efficiency

---

## Contact & Resources

### Key Files
- **Training Script**: `scripts/train.py` (fixed)
- **Main README**: `README.md` (updated)
- **Archive**: `archive/modelscope_3.2M_results/` (historical)
- **This Status**: `CURRENT_STATUS.md` (you are here)

### Quick Commands
```bash
# Run 500-step baseline training
python scripts/train.py --config configs/train_cpu.yaml --steps 500 --limit 1000 --chunk_size 10

# Run 1000+ step extended training (recommended for production)
python scripts/train.py --config configs/train_cpu.yaml --steps 1000 --limit 32 --chunk_size 2

# Run evaluation with proper config dimensions
python scripts/eval.py --config configs/train_cpu.yaml --limit 32 --chunk_size 2

# Run scaling experiment
python scripts/train.py --config configs/train_scaling.yaml

# Build KG
python scripts/build_kg.py

# Run tests
pytest tests/
```

---

**Last Updated**: 2024  
**Status**: ✅ Bug fixed, ready for baseline establishment  
**Next Milestone**: 500-step training with 85M model  
**Long-term Goal**: Production deployment with full 29,174 triple dataset

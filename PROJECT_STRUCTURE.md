# GraphMER-SE Project Structure

## 📁 Repository Organization

```
graphMER/
├── README.md                           # Main project documentation
├── CURRENT_STATUS.md                   # Current implementation status
├── PRODUCTION_CHECKLIST.md             # Production readiness checklist
├── PRODUCTION_RESULTS.md               # 5,000-step training results
├── HIGH_PRIORITY_COMPLETE.md           # Advanced features implementation
├── AUDIT_REPORT.md                     # Comprehensive implementation audit
├── CONTRIBUTING.md                     # Contribution guidelines
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project configuration
├── pytest.ini                         # Test configuration
├── Dockerfile                          # Container setup
├── Makefile                           # Build automation
│
├── src/                               # Source code
│   ├── models/                        # Model architecture
│   │   ├── encoder.py                 # 85M parameter transformer encoder
│   │   └── heads.py                   # MLM and MNM prediction heads
│   ├── training/                      # Training components
│   │   ├── dataset_v2.py              # Leafy Chain dataset builder
│   │   ├── tokenizer_bpe.py           # Code-aware BPE tokenizer
│   │   └── constraint_loss.py         # Ontology constraint regularizers
│   ├── ontology/                      # Knowledge graph validation
│   │   ├── validator.py               # Structural validation
│   │   └── kg_validator.py            # Constraint checking
│   └── utils/                         # Utility functions
│       └── metrics.py                 # Training metrics
│
├── scripts/                           # Training and evaluation scripts
│   ├── train_v2.py                    # Production training with advanced features
│   ├── run_gpu_profile.py             # GPU profile runner
│   ├── build_kg_enhanced.py           # Knowledge graph builder
│   ├── eval_comprehensive.py          # Full evaluation suite
│   ├── validate_production.py         # Production validation
│   └── repro_harness.py               # Reproducibility testing
│
├── configs/                           # Configuration files
│   ├── train_v2_gpu.yaml             # GPU training with all features
│   ├── train_cpu.yaml                 # CPU training fallback
│   ├── gpu_profiles.yaml             # Validated GPU configurations
│   └── train_scaling.yaml            # Long-run scaling config
│
├── data/                              # Data and knowledge graphs
│   ├── kg/                            # Knowledge graph files
│   │   ├── seed_python.jsonl         # Production KG (21k+ triples)
│   │   └── manifest.json             # KG metadata and validation
│   ├── tokenizer/                     # Tokenizer files
│   │   └── code_bpe.json             # BPE vocabulary (8000 tokens)
│   └── raw/                           # Raw source code samples
│       └── python_samples/            # Python code for KG building
│
├── logs/                              # Training logs and checkpoints
│   ├── checkpoints/                   # Model checkpoints
│   ├── train_v2_metrics.csv          # Training metrics
│   └── evaluation_results.json       # Evaluation results
│
├── tests/                             # Test suite
│   ├── test_model.py                  # Model architecture tests
│   ├── test_dataset.py               # Dataset functionality tests
│   └── test_metadata.py              # Metadata validation tests
│
└── docs/                              # Additional documentation
    ├── specs/                         # Specifications
    │   ├── ontology_spec.yaml         # SE ontology definition
    │   └── data_spec.yaml             # Data governance
    └── model_card.md                  # Model specifications
```

## 🔧 Key Components

### Core Implementation
- **`src/models/encoder.py`** - 85M parameter transformer with relation-aware attention
- **`src/training/dataset_v2.py`** - Leafy Chain encoding with dual MLM+MNM objectives
- **`src/training/constraint_loss.py`** - Ontology constraint regularizers (NEW)
- **`scripts/train_v2.py`** - Production training loop with all advanced features

### Configuration Management
- **`configs/train_v2_gpu.yaml`** - GPU training with constraint regularizers and curriculum learning
- **`configs/gpu_profiles.yaml`** - Validated 8GB/16GB GPU configurations
- **`data/kg/manifest.json`** - Knowledge graph metadata with validation results

### Production Infrastructure
- **`scripts/run_gpu_profile.py`** - Validated GPU training profiles
- **`scripts/validate_production.py`** - Production readiness validation
- **`scripts/repro_harness.py`** - Reproducibility testing with streaming output

## 📊 Data Flow

1. **Knowledge Graph Building**: `build_kg_enhanced.py` → `data/kg/seed_python.jsonl`
2. **Dataset Creation**: `dataset_v2.py` processes KG + code → Leafy Chain format
3. **Training**: `train_v2.py` with constraint regularizers → model checkpoints
4. **Evaluation**: `eval_comprehensive.py` → performance metrics
5. **Validation**: `validate_production.py` → production readiness assessment

## 🚀 Quick Navigation

- **Start Training**: `scripts/run_gpu_profile.py --profile 408032G`
- **Build KG**: `scripts/build_kg_enhanced.py --source_dir data/raw/python_samples`
- **Run Evaluation**: `scripts/eval_comprehensive.py --checkpoint logs/checkpoints/model_v2_*.pt`
- **Check Status**: `CURRENT_STATUS.md` and `PRODUCTION_RESULTS.md`
- **Implementation Details**: `HIGH_PRIORITY_COMPLETE.md` and `AUDIT_REPORT.md`

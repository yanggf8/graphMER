# ModelScope Training Guide for GraphMER-SE

This document provides detailed instructions for training the GraphMER-SE model using ModelScope resources.

## Prerequisites

1. **ModelScope Account** - Register at https://modelscope.cn (no credit card required)
2. **ModelScope SDK Installation**:
```bash
pip install modelscope
```

## Setup and Installation

First, install the required dependencies:

```bash
# Install GraphMER-SE in development mode
pip install -e .

# Install additional dependencies for ModelScope
pip install modelscope transformers datasets
```

## Prepare Your Data

Before training, you need to build your knowledge graph dataset:

```bash
# Build knowledge graph from your codebase
python scripts/build_kg.py

# Or use the enhanced builder for larger datasets
python scripts/build_kg_enhanced.py --source_dir data/raw/python_samples
```

## Training with ModelScope

### Method 1: Using ModelScope-Compatible Script

To run training with ModelScope optimizations:

```bash
python scripts/train.py --config configs/modelscope_config.yaml --steps 100 --modelscope
```

### Method 2: Using Dedicated ModelScope Training Script

For full ModelScope integration, you can use the dedicated training script:

```bash
python train_modelscope.py
```

This script:
- Adapts the GraphMER model architecture for ModelScope
- Handles dataset conversion to ModelScope format
- Manages training with appropriate resource constraints

### Method 3: Using ModelScope Studio (Recommended)

1. Go to ModelScope Studio: https://modelscope.cn/studios
2. Create a new notebook environment
3. Clone the GraphMER repository:

```bash
!git clone https://github.com/your-repo/graphMER.git
%cd graphMER

# Install dependencies
!pip install -e .
!pip install modelscope transformers datasets

# Build knowledge graph
!python scripts/build_kg_enhanced.py --source_dir data/raw/python_samples

# Run training
!python train_modelscope.py
```

## ModelScope Configuration Options

The `configs/modelscope_config.yaml` file is optimized for ModelScope resources with:

- Reduced model dimensions for memory efficiency
- Adjusted learning rate for faster convergence
- Curriculum learning for better stability
- Activation checkpointing to save memory

## Custom Configuration for ModelScope

You can modify the configuration by editing `configs/modelscope_config.yaml`:

- `hardware.device`: Set to `auto` for automatic GPU detection
- `model.hidden_size`: Reduce for memory constraints
- `training_data.max_seq_len`: Adjust based on memory availability
- `run.epochs`: Limit for time-bounded training sessions

## Resource Optimization Tips

1. **Memory Management**: Use activation checkpointing to reduce memory usage
2. **Batch Size**: Start with smaller batch sizes and increase as needed
3. **Sequence Length**: Use curriculum learning to gradually increase sequence lengths
4. **Learning Rate**: Higher learning rates can help with faster convergence in limited time sessions

## Monitoring Training Progress

Training metrics are saved to `logs/train_metrics.csv` and include:
- Total loss
- MLM loss and accuracy
- MNM loss and accuracy
- Validation metrics

## Troubleshooting

**Common Issues:**

1. **Memory Issues**: Reduce `max_seq_len` or `micro_batch_size` in config
2. **Session Timeouts**: Implement checkpointing to resume training
3. **Package Conflicts**: Use `pip install --no-deps` for specific versions

**ModelScope-Specific Issues:**

- Check ModelScope documentation: https://modelscope.cn/docs
- Verify GPU availability: `nvidia-smi`
- Monitor resource usage during training

## Validated Training Results

### 500-Step Training (Completed)

✅ **Successfully validated** on ModelScope-optimized configuration:

- **Total Loss Reduction**: 44.7% (0.3754 → 0.2076)
- **MLM Loss Reduction**: 60.9% (5.9779 → 2.3399)
- **MNM Loss Reduction**: 28.7% (6.0338 → 4.3040)
- **Peak MLM Accuracy**: 81.82%
- **Peak MNM Accuracy**: 32.26%

**Dataset**: 100 samples from 29,174 available triples  
**Configuration**: `configs/modelscope_config.yaml`  
**Details**: See `docs/MODELSCOPE_500STEP_RESULTS.md`

### Recommended Training Command

For scaled-up training with the full dataset:

```bash
# Run 1000 steps with larger sample size
python scripts/train.py \
  --config configs/modelscope_config.yaml \
  --steps 1000 \
  --limit 5000 \
  --chunk_size 50
```

## Best Practices

1. **Start Small**: Begin with a small model configuration and gradually scale up
2. **Monitor Metrics**: Keep an eye on loss curves and validation accuracy
3. **Save Checkpoints**: Regularly save model checkpoints for resuming training
4. **Validate Results**: Use the evaluation script to verify model performance
5. **Scale Gradually**: Increase dataset size and training steps incrementally (100 → 500 → 1000+)

## Integration with Existing Workflow

The ModelScope training integrates with the existing workflow:

- Uses the same knowledge graph building process
- Compatible with existing evaluation scripts
- Maintains the same model architecture
- Preserves ablation tracking functionality

For more information about the base GraphMER-SE model and its capabilities, see the main README.md file.
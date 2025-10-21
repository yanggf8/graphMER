# ModelScope Training Guide for GraphMER-SE

This guide explains how to train GraphMER-SE using ModelScope's free training resources.

## Reference
- ModelScope Training Documentation: https://modelscope.cn/docs/sdk/model-training

## Prerequisites

1. **ModelScope Account** - Register at https://modelscope.cn (no credit card required)
2. **ModelScope SDK Installation**:
```bash
pip install modelscope
```

## Setup for GraphMER-SE

### 1. Prepare Training Environment

```python
from modelscope.trainers import build_trainer
from modelscope.msdatasets import MsDataset
from modelscope.utils.config import Config

# Install GraphMER-SE dependencies
!pip install torch transformers datasets
!git clone https://github.com/your-repo/graphMER.git
%cd graphMER
!pip install -e .
```

### 2. Adapt GraphMER-SE for ModelScope

Create `modelscope_config.py`:

```python
from modelscope.utils.config import Config

# ModelScope training configuration
config = Config({
    'task': 'text-classification',  # Adapt based on your task
    'model': {
        'type': 'GraphMEREncoder',
        'model_dir': './src/models/',
        'num_labels': 2,  # Adjust for your task
    },
    'dataset': {
        'train_dataset': {
            'type': 'custom',
            'data_dir': './data/kg/',
        }
    },
    'train': {
        'max_epochs': 10,
        'batch_size': 8,
        'lr': 2e-5,
        'save_checkpoint_epochs': 2,
    }
})
```

### 3. Create ModelScope-Compatible Dataset

```python
# Convert GraphMER-SE data to ModelScope format
import json
from modelscope.msdatasets import MsDataset

def prepare_modelscope_dataset():
    # Load your KG triples and training data
    with open('data/kg/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Convert to ModelScope dataset format
    train_data = []
    # Add your data conversion logic here
    
    # Save in ModelScope format
    dataset = MsDataset.from_dict({
        'train': train_data
    })
    return dataset
```

### 4. Training Script for ModelScope

Create `train_modelscope.py`:

```python
from modelscope.trainers import build_trainer
from modelscope.utils.config import Config
import sys
sys.path.append('./src')

from models.graphmer_encoder import GraphMEREncoder

def train_on_modelscope():
    # Load config
    config = Config.from_file('modelscope_config.py')
    
    # Build trainer
    trainer = build_trainer(
        name='text-classification-trainer',
        model=GraphMEREncoder,
        cfg=config
    )
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    train_on_modelscope()
```

## ModelScope Notebook Integration

### 1. Create ModelScope Notebook

1. Go to ModelScope Studio: https://modelscope.cn/studios
2. Create new notebook
3. Select GPU environment (free tier available)

### 2. Upload GraphMER-SE Code

```bash
# In ModelScope notebook
!git clone https://github.com/your-repo/graphMER.git
%cd graphMER

# Build KG
!python scripts/build_kg_enhanced.py --source_dir data/raw/python_samples

# Install dependencies
!pip install -e .
```

### 3. Run Training

```python
# Adapt existing training script
!python train_modelscope.py
```

## Resource Limits

**ModelScope Free Tier:**
- GPU hours: Limited per day/week
- Storage: Temporary (save checkpoints to external storage)
- Memory: Varies by instance type

**Optimization for GraphMER-SE:**
- Use CPU config initially: `configs/train_cpu.yaml`
- Enable mixed precision for GPU: `bf16: true`
- Checkpoint frequently due to session limits

## Integration with Existing Workflow

### Minimal Changes Required

1. **Keep existing configs**: Use `configs/train_cpu.yaml` as base
2. **Adapt data loading**: Convert KG triples to ModelScope format
3. **Checkpoint management**: Save to ModelScope's persistent storage

### Example Integration

```python
# In your existing train.py, add ModelScope support
def train_with_modelscope():
    if os.getenv('MODELSCOPE_ENV'):
        # Use ModelScope trainer
        from modelscope.trainers import build_trainer
        trainer = build_trainer(name='custom', cfg=config)
        trainer.train()
    else:
        # Use existing training loop
        train_graphmer_se()
```

## Next Steps

1. Register ModelScope account
2. Test notebook environment with small dataset
3. Adapt GraphMER-SE training script
4. Run production training with full 30k+ triples

## Troubleshooting

**Common Issues:**
- Package conflicts: Use `!pip install --no-deps` for specific versions
- Memory limits: Reduce batch size in config
- Session timeouts: Implement checkpoint resumption

**ModelScope Support:**
- Documentation: https://modelscope.cn/docs
- Community: ModelScope forums and GitHub issues

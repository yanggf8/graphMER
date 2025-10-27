# Checkpoint Management

## Current Checkpoint

**Latest Model**: `logs/checkpoints/model_v2_20251027_171135_s42.pt`
- **Size**: 1.3GB
- **Training Steps**: 1,000
- **Loss**: 6.999 (57% reduction)
- **Features**: Full GraphMER compliance with all advanced features

## Automatic Cleanup

The training script automatically manages checkpoints:

```python
# Keeps only latest 2 checkpoints during training
checkpoints = sorted(checkpoint_dir.glob("model_v2_step*.pt"), key=lambda x: x.stat().st_mtime)
for old_cp in checkpoints[:-2]:
    old_cp.unlink()
```

## Manual Cleanup

To clean up old checkpoints manually:

```bash
# Keep only the latest final checkpoint
cd /home/yanggf/a/graphMER
ls -t logs/checkpoints/model_v2_*_s42.pt | head -1 > /tmp/keep_checkpoint
KEEP_FILE=$(cat /tmp/keep_checkpoint)

# Remove all other large checkpoints
find logs/checkpoints -name "*.pt" -size +100M ! -path "$KEEP_FILE" -delete
```

## Checkpoint Contents

Each checkpoint contains:
- `model_state_dict`: TinyEncoder parameters
- `mlm_head_state_dict`: MLM prediction head
- `mnm_head_state_dict`: MNM prediction head  
- `optimizer_state_dict`: Optimizer state
- `config`: Training configuration
- `step`: Training step number
- `loss`: Final loss value

## Loading Checkpoints

```python
import torch
from src.models.encoder import TinyEncoder

# Load checkpoint
checkpoint = torch.load('logs/checkpoints/model_v2_20251027_171135_s42.pt')

# Restore model
config = checkpoint['config']['model']
model = TinyEncoder(**config)
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Loaded model from step {checkpoint['step']} with loss {checkpoint['loss']:.3f}")
```

## Storage Recommendations

- **Development**: Keep 1-2 latest checkpoints (~2.6GB)
- **Production**: Archive final checkpoint, remove intermediates
- **Research**: Keep checkpoints at key milestones (1k, 5k, 10k steps)

## Disk Usage

- **Before Cleanup**: ~8GB (multiple large checkpoints)
- **After Cleanup**: ~1.3GB (single final checkpoint)
- **Training**: Max 2.6GB (2 checkpoints during training)

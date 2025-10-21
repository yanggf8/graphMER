# GraphMER-SE: Upload to Google Drive Instructions

**Package Ready:** `graphmer_colab.tar.gz` (631 KB)  
**Date:** October 20, 2025  
**Status:** ✅ Ready for upload

---

## Package Contents

Your Colab deployment package includes:

```
graphmer_colab.tar.gz (631 KB)
├── src/                    # Source code (model, data, ontology)
├── data/                   # Knowledge graph data
│   ├── enhanced_multilang.jsonl           (6.6M - 30,826 triples)
│   ├── enhanced_multilang.entities.jsonl  (510K)
│   └── manifest.json                      (24K - build metadata)
├── configs/
│   ├── train_tpu.yaml     # Original TPU config
│   └── train_colab.yaml   # Colab-optimized config ⭐ NEW
├── scripts/               # Training and validation scripts
├── docs/specs/
│   └── ontology_spec.yaml # Ontology specification
├── pyproject.toml         # Python dependencies
└── COLAB_TPU_SETUP.md     # Complete setup guide
```

**Total Size:** 631 KB compressed (16 MB uncompressed)

---

## Step 1: Upload to Google Drive

### Option A: Manual Upload (Recommended - Easiest)

1. **Go to Google Drive:**
   - Open https://drive.google.com
   - Log in with your Google account

2. **Create Project Folder:**
   - Click "New" → "Folder"
   - Name it: `GraphMER`
   - Open the folder

3. **Upload Package:**
   - Drag and drop `graphmer_colab.tar.gz` into the GraphMER folder
   - OR click "New" → "File upload" → Select `graphmer_colab.tar.gz`
   - Wait for upload to complete (~30 seconds for 631 KB)

4. **Verify Upload:**
   - You should see: `graphmer_colab.tar.gz` (631 KB) in the GraphMER folder
   - Right-click → "Get link" → Copy link (optional, for sharing)

### Option B: Using `rclone` (Advanced)

If you have `rclone` configured for Google Drive:

```bash
# Upload to Google Drive
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/

# Verify upload
rclone ls gdrive:GraphMER/
```

### Option C: Using Google Drive Desktop App

If you have Google Drive Desktop installed:

1. Copy `graphmer_colab.tar.gz` to your local Google Drive folder
2. Wait for sync to complete
3. Verify on drive.google.com

---

## Step 2: Create Google Colab Notebook

1. **Go to Google Colab:**
   - Open https://colab.research.google.com
   - Log in with the same Google account

2. **Create New Notebook:**
   - Click "File" → "New notebook"
   - Rename to: `GraphMER_Training.ipynb`

3. **Change Runtime to TPU:**
   - Click "Runtime" → "Change runtime type"
   - Select "TPU" under Hardware accelerator
   - Click "Save"

4. **Verify TPU Access:**
   - Add this cell and run it:
   ```python
   # Cell 1: Verify TPU
   import torch_xla.core.xla_model as xm
   print(f"✅ TPU cores: {xm.xrt_world_size()}")
   # Expected output: ✅ TPU cores: 8
   ```

---

## Step 3: Copy Colab Notebook Template

Copy these cells into your Colab notebook:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted")
```

### Cell 2: Extract Project
```python
import os

# Extract package
!tar -xzf /content/drive/MyDrive/GraphMER/graphmer_colab.tar.gz -C /content/
print("✅ Project extracted")

# Change to project directory
os.chdir('/content/colab_deploy')
!pwd
```

### Cell 3: Verify Data
```python
# Check data files
!ls -lh data/
!wc -l data/enhanced_multilang.jsonl
# Expected: 30826 data/enhanced_multilang.jsonl
print("✅ Data verified")
```

### Cell 4: Install Dependencies
```python
# Install required packages
!pip install -q transformers datasets pyyaml networkx torch

print("✅ Dependencies installed")
```

### Cell 5: Run Data Validation
```python
# Validate knowledge graph quality
!python src/ontology/kg_validator.py \
  data/enhanced_multilang.jsonl \
  data/enhanced_multilang.entities.jsonl \
  docs/specs/ontology_spec.yaml

print("✅ Data validation complete")
# Expected: domain_range_ratio: 0.991, inherits_acyclic: True
```

### Cell 6: Check TPU Status
```python
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"✅ TPU device: {device}")
print(f"✅ TPU cores: {xm.xrt_world_size()}")
print(f"✅ torch-xla version: {torch_xla.__version__}")
```

### Cell 7: Create Output Directories
```python
# Create directories for outputs
!mkdir -p /content/drive/MyDrive/GraphMER/outputs
!mkdir -p /content/drive/MyDrive/GraphMER/checkpoints
print("✅ Output directories created")
```

### Cell 8: Run Short Training Test (5 minutes)
```python
# Run 100-step smoke test
!python scripts/train.py \
  --config configs/train_colab.yaml \
  --steps 100 \
  --output_dir /content/drive/MyDrive/GraphMER/outputs

print("✅ Training test complete")
```

### Cell 9: Verify Training Metrics
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('/content/drive/MyDrive/GraphMER/outputs/train_metrics.csv')

# Display last 10 rows
print("\n📊 Last 10 training steps:")
print(df.tail(10))

# Plot metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(df['step'], df['train_loss'])
ax1.set_title('Training Loss')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(df['step'], df['val_acc'])
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Step')
ax2.set_ylabel('Accuracy')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("✅ Metrics verified")
```

### Cell 10: Full Training Run (Optional - 15-20 min per 1000 steps)
```python
# Run full training (adjust steps as needed)
!python scripts/train.py \
  --config configs/train_colab.yaml \
  --steps 1000 \
  --output_dir /content/drive/MyDrive/GraphMER/outputs \
  --checkpoint_dir /content/drive/MyDrive/GraphMER/checkpoints

print("✅ Full training complete")
```

### Cell 11: Resume from Checkpoint (if session expires)
```python
# Resume training from last checkpoint
!python scripts/train.py \
  --config configs/train_colab.yaml \
  --resume_from /content/drive/MyDrive/GraphMER/checkpoints/checkpoint_1000.pt \
  --steps 2000 \
  --output_dir /content/drive/MyDrive/GraphMER/outputs

print("✅ Resumed training complete")
```

---

## Step 4: Save Notebook to Google Drive

1. **Save notebook:**
   - Click "File" → "Save a copy in Drive"
   - Save to: `My Drive/GraphMER/GraphMER_Training.ipynb`

2. **Share notebook (optional):**
   - Click "Share" button
   - Copy link to share with collaborators

---

## Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Upload to Drive | ~30 sec | 631 KB file |
| Create notebook | 2 min | Including TPU runtime setup |
| Extract & verify | 1 min | Cells 1-5 |
| Smoke test (100 steps) | 5 min | Cell 8 |
| Training (1000 steps) | 15-20 min | Cell 10 |
| Training (5000 steps) | 75-100 min | Adjust steps in Cell 10 |
| Full training (50k steps) | 12-16 hours | Multiple sessions with checkpoints |

---

## Troubleshooting

### Issue: "File not found" when extracting
**Solution:**
```python
# Check file exists
!ls -lh /content/drive/MyDrive/GraphMER/
# If not found, verify upload completed and path is correct
```

### Issue: "TPU not found"
**Solution:**
- Runtime → Change runtime type → TPU → Save
- Runtime → Restart runtime
- Re-run Cell 1 (verify TPU)

### Issue: "Session disconnected after 12 hours"
**Solution:**
- Normal behavior for Colab Free
- Checkpoints are saved to Drive every 500 steps
- Use Cell 11 to resume from last checkpoint
- Consider Colab Pro ($10/month) for 24h sessions

### Issue: "Out of memory"
**Solution:**
Reduce batch size in `configs/train_colab.yaml`:
```yaml
training_data:
  micro_batch_size: 1  # Reduce from 2
```

---

## Next Steps After Upload

1. ✅ **Package uploaded to Drive** (you are here after upload)
2. ⏳ **Create Colab notebook** with cells above
3. ⏳ **Run validation** (Cells 1-5, ~2 min)
4. ⏳ **Run smoke test** (Cell 8, ~5 min)
5. ⏳ **Run full training** (Cell 10, adjust steps as needed)
6. ⏳ **Monitor metrics** (Cell 9)
7. ⏳ **Archive results** (save checkpoints and metrics to Drive)

---

## File Locations Reference

**In this directory (WSL):**
- `graphmer_colab.tar.gz` - Package to upload

**In Google Drive (after upload):**
- `My Drive/GraphMER/graphmer_colab.tar.gz`

**In Colab (after extraction):**
- `/content/colab_deploy/` - Project directory
- `/content/colab_deploy/data/enhanced_multilang.jsonl` - KG data
- `/content/colab_deploy/configs/train_colab.yaml` - Config

**Output Locations:**
- `/content/drive/MyDrive/GraphMER/outputs/` - Training logs & metrics
- `/content/drive/MyDrive/GraphMER/checkpoints/` - Model checkpoints

---

## Summary

✅ **Package ready:** `graphmer_colab.tar.gz` (631 KB)  
✅ **Contents validated:** 30,826 triples, 99.10% quality  
✅ **Upload destination:** Google Drive → GraphMER folder  
✅ **Next step:** Upload to Drive, then create Colab notebook

**You're ready to deploy! 🚀**

---

**Created:** 2025-10-20  
**Package size:** 631 KB  
**Upload time:** ~30 seconds  
**Status:** Ready for Google Colab TPU training

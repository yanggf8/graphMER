# Kaggle Training Notebook - Setup Instructions

**Dataset uploaded:** ✅ https://www.kaggle.com/datasets/yanggoufang/graphmer-kg
**Status:** Ready to train!

---

## Quick Setup (5 minutes)

### Step 1: Create New Notebook

1. **Go to:** https://www.kaggle.com/code
2. **Click:** "New Notebook" button (top right)

---

### Step 2: Configure Settings

Click **"Settings"** (right sidebar):

1. **Accelerator:**
   - Select: **GPU**
   - Click: **Save**

2. **Internet:**
   - Toggle: **On**

3. **Persistence:**
   - (Optional) Enable if you want to save variables between sessions

---

### Step 3: Add Your Dataset

1. **Click "Add Data"** button (right sidebar)
2. **Search:** `graphmer-kg` or `yanggoufang/graphmer-kg`
3. **Click "Add"** next to your dataset
4. **Verify:** Dataset appears in sidebar under "Input"

**Dataset will mount at:** `/kaggle/input/graphmer-kg/`

---

### Step 4: Import Notebook (Option A - Recommended)

**Upload the pre-built notebook:**

1. **File → Import Notebook**
2. **Upload:** `/home/yanggf/a/graphMER/GraphMER_Kaggle_Training.ipynb`
3. **Wait for upload** (a few seconds)
4. **Run Cell 1** to verify GPU

**That's it!** The notebook has all 14 cells ready to go.

---

### Step 4: Manual Setup (Option B - If Import Doesn't Work)

If you can't import the notebook, create cells manually:

#### Cell 1: Verify GPU

\`\`\`python
import torch
import subprocess

# Verify GPU
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available! Go to Settings → GPU → Save")

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"✅ GPU: {gpu_name}")
print(f"✅ VRAM: {gpu_memory:.1f} GB")
print(f"✅ CUDA: {torch.version.cuda}")

subprocess.run(['nvidia-smi'])
\`\`\`

#### Cell 2: Verify Dataset

\`\`\`python
import os
import subprocess

dataset_path = '/kaggle/input/graphmer-kg'

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found! Add it via 'Add Data' button")

print("📊 Dataset files:")
subprocess.run(['ls', '-lh', dataset_path])

# Verify triple count
triples_file = f"{dataset_path}/enhanced_multilang.jsonl"
result = subprocess.run(['wc', '-l', triples_file], capture_output=True, text=True)
count = int(result.stdout.split()[0])

print(f"\n✅ Data verified: {count:,} triples")
\`\`\`

#### Cell 3: Setup Working Directory

\`\`\`python
import os
import shutil

# Create working directories
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/outputs', exist_ok=True)

# Copy source code from dataset to working directory
for item in ['src', 'configs', 'scripts']:
    src_zip = f'/kaggle/input/graphmer-kg/{item}.zip'
    if os.path.exists(src_zip):
        subprocess.run(['unzip', '-q', src_zip, '-d', '/kaggle/working/'])
        print(f"✅ Extracted {item}")

os.chdir('/kaggle/working')
print(f"\n✅ Working directory: {os.getcwd()}")
\`\`\`

#### Cell 4: Install Dependencies

\`\`\`python
%%time
print("📦 Installing dependencies...")

!pip install -q transformers datasets pyyaml networkx tensorboard

import transformers
import datasets
import yaml

print("\n✅ Dependencies installed:")
print(f"   • transformers: {transformers.__version__}")
print(f"   • datasets: {datasets.__version__}")
\`\`\`

#### Cell 5: Smoke Test (100 steps)

\`\`\`python
%%time
print("🚀 Starting smoke test (100 steps, ~30 seconds)...")

!python scripts/train.py \\
  --config configs/train_kaggle.yaml \\
  --steps 100 \\
  --output_dir /kaggle/working/outputs

print("\n✅ Smoke test complete!")
\`\`\`

#### Cell 6: Full Training (10k steps)

\`\`\`python
%%time
print("🚀 Starting full training (10,000 steps, ~10-15 minutes)...")

!python scripts/train.py \\
  --config configs/train_kaggle.yaml \\
  --steps 10000 \\
  --output_dir /kaggle/working/outputs \\
  --checkpoint_dir /kaggle/working/checkpoints

print("\n✅ Training complete!")
\`\`\`

---

## Expected Results

### After Smoke Test (Cell 5)
```
✅ GPU detected: Tesla T4 or P100
✅ Data loaded: 29,174 triples
✅ Training completed: 100 steps
⏱️ Time: ~30 seconds
```

### After Full Training (Cell 6)
```
✅ Training completed: 10,000 steps
✅ Checkpoints saved: Every 1000 steps
⏱️ Time: ~10-15 minutes
💾 GPU hours used: ~0.2h / 30h weekly quota
```

---

## Troubleshooting

### Issue: GPU not available
**Solution:**
- Settings → Accelerator → GPU → Save
- Runtime → Restart Session

### Issue: Dataset not found
**Solution:**
- Click "Add Data" → Search "graphmer-kg" → Add
- Check path: `!ls /kaggle/input/`

### Issue: Out of memory
**Solution:**
Edit `/kaggle/working/configs/train_kaggle.yaml`:
```yaml
training_data:
  micro_batch_size: 32  # Reduce from 64
```

### Issue: Files not extracting
**Solution:**
```python
# Manual extraction
!unzip -q /kaggle/input/graphmer-kg/src.zip -d /kaggle/working/
!unzip -q /kaggle/input/graphmer-kg/configs.zip -d /kaggle/working/
!unzip -q /kaggle/input/graphmer-kg/scripts.zip -d /kaggle/working/
```

---

## Performance Tracking

**GPU Quota Usage:**
- Smoke test (100 steps): ~0.01h
- Full training (10k steps): ~0.2h
- Weekly quota: 30h total
- Remaining after 10k: ~29.8h

**Training Capacity:**
- 30 GPU hours = ~1.5M training steps
- Or ~150 runs of 10k steps each

---

## Next Steps After Training

1. **Download outputs:**
   ```python
   import tarfile
   with tarfile.open('outputs.tar.gz', 'w:gz') as tar:
       tar.add('/kaggle/working/checkpoints')
       tar.add('/kaggle/working/train_metrics.csv')
   ```
   Then: Right-click in sidebar → Download

2. **View metrics:**
   - Check `/kaggle/working/train_metrics.csv`
   - Plot loss and accuracy curves

3. **Continue training:**
   - Resume from checkpoint
   - Adjust hyperparameters
   - Run longer (50k, 100k steps)

---

## Quick Reference

**Dataset path:** `/kaggle/input/graphmer-kg/`
**Working path:** `/kaggle/working/`
**Config file:** `/kaggle/working/configs/train_kaggle.yaml`
**Training script:** `/kaggle/working/scripts/train.py`

**Data files:**
- Triples: `enhanced_multilang.jsonl` (29,174 lines)
- Entities: `enhanced_multilang.entities.jsonl` (9,375 lines)

**Key settings:**
- Batch size: 64
- Mixed precision: FP16
- Checkpoint interval: 1000 steps
- GPU: Tesla T4 or P100 (16GB VRAM)

---

## Complete Workflow Summary

```
1. Create notebook        → kaggle.com/code
2. Enable GPU            → Settings → GPU → Save
3. Add dataset           → Add Data → graphmer-kg
4. Import notebook       → File → Import → GraphMER_Kaggle_Training.ipynb
5. Run smoke test        → Cell 1-5 (30 sec)
6. Run full training     → Cell 6 (10-15 min)
7. Download results      → Package and download
```

---

**Ready to start!** 🚀

Your next action: Go to https://www.kaggle.com/code and create a new notebook.

**Files you'll need:**
- Pre-built notebook: `/home/yanggf/a/graphMER/GraphMER_Kaggle_Training.ipynb`
- Or use manual cells above

---

**Created:** 2025-10-21
**Dataset:** yanggoufang/graphmer-kg (29,174 triples)
**Status:** ✅ Production ready

# GraphMER-SE: Kaggle Upload Quick Guide

**Goal:** Get GraphMER training on Kaggle in 10 minutes

---

## Step 1: Build Dataset Package (2 minutes)

```bash
cd /home/yanggf/a/graphMER
./scripts/deploy_to_kaggle.sh
```

**Output:** `graphmer_kaggle_dataset.zip` (~3-5 MB)

**What's included:**
- ✅ 30,826 knowledge graph triples
- ✅ Source code (src/)
- ✅ Configs (train_kaggle.yaml optimized for Kaggle GPU)
- ✅ Scripts (train.py, eval.py)
- ✅ Documentation (README, ontology spec)

---

## Step 2: Upload Dataset to Kaggle (3 minutes)

### Option A: Web Upload (Easiest)

1. **Go to:** https://www.kaggle.com/datasets

2. **Click:** "New Dataset" button

3. **Upload:**
   - Drag & drop: `graphmer_kaggle_dataset.zip`
   - Or click "Upload Files" and select the file

4. **Fill in:**
   - **Title:** `GraphMER Knowledge Graph`
   - **Subtitle:** `High-quality KG for software engineering (30,826 triples)`
   - **Tags:** `knowledge graph`, `nlp`, `software engineering`
   - **Visibility:** Private (recommended) or Public

5. **Click:** "Create"

6. **Note:** Your dataset URL will be:
   ```
   https://www.kaggle.com/datasets/YOUR_USERNAME/graphmer-knowledge-graph
   ```

### Option B: Kaggle CLI (Alternative)

```bash
# Install CLI
pip install kaggle

# Get API token
# 1. Go to: https://www.kaggle.com/YOUR_USERNAME/account
# 2. Click: "Create New API Token"
# 3. Save: kaggle.json to ~/.kaggle/

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Upload dataset
cd /home/yanggf/a/graphMER/kaggle_deploy
nano dataset-metadata.json  # Update YOUR_USERNAME
kaggle datasets create -p .
```

---

## Step 3: Create Training Notebook (3 minutes)

### Method 1: Upload Pre-built Notebook (Fastest)

1. **Go to:** https://www.kaggle.com/code

2. **Click:** "New Notebook"

3. **Import:**
   - File → Import Notebook
   - Upload: `/home/yanggf/a/graphMER/GraphMER_Kaggle_Training.ipynb`

4. **Configure:**
   - Settings → Accelerator → **GPU** → Save
   - Settings → Internet → **On**

5. **Add Dataset:**
   - Click "Add Data" (right sidebar)
   - Search: `graphmer-knowledge-graph`
   - Click "Add"

6. **Verify:**
   - Run Cell 1: Check GPU
   - Run Cell 2: Verify dataset mounted at `/kaggle/input/graphmer-knowledge-graph/`

### Method 2: Copy-Paste Cells (Alternative)

Create new notebook and paste these cells:

**Cell 1: Verify GPU**
```python
import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available!")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
```

**Cell 2: Setup**
```python
import os
import shutil

# Copy source from dataset
for item in ['src', 'configs', 'scripts']:
    src = f'/kaggle/input/graphmer-knowledge-graph/{item}'
    dst = f'/kaggle/working/{item}'
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

os.chdir('/kaggle/working')
print(f"✅ Working directory: {os.getcwd()}")
```

**Cell 3: Install Dependencies**
```python
!pip install -q transformers datasets pyyaml networkx tensorboard
print("✅ Dependencies installed")
```

**Cell 4: Smoke Test**
```python
%%time
!python scripts/train.py \
  --config configs/train_kaggle.yaml \
  --steps 100 \
  --output_dir /kaggle/working/outputs
```

**Cell 5: Full Training**
```python
%%time
!python scripts/train.py \
  --config configs/train_kaggle.yaml \
  --steps 10000 \
  --output_dir /kaggle/working/outputs \
  --checkpoint_dir /kaggle/working/checkpoints
```

---

## Step 4: Run Training (2 minutes)

### Smoke Test (30 seconds)

1. **Run Cells 1-4** (setup + smoke test)

**Expected output:**
```
✅ GPU: Tesla T4 (16GB)
✅ Data loaded: 30,826 triples
Step 100/100 | Loss: 0.234 | 15 steps/sec
✅ Training complete!
Time: ~30 seconds
```

### Full Training (10-15 minutes)

2. **Run Cell 5** (10,000 steps)

**Expected output:**
```
Step 10000/10000 | Loss: 0.089 | Acc: 0.912
✅ Training complete!
Time: ~10-15 minutes
```

---

## Step 5: Download Results (2 minutes)

**Option A: Direct Download**

```python
# Add this cell
import tarfile
with tarfile.open('outputs.tar.gz', 'w:gz') as tar:
    tar.add('/kaggle/working/checkpoints')
    tar.add('/kaggle/working/train_metrics.csv')
```

Then: Right-click `outputs.tar.gz` in sidebar → Download

**Option B: Save as Kaggle Dataset**

1. Click: "Save Version" → "Quick Save"
2. Navigate to: "Data" tab
3. Click: "New Dataset" from outputs
4. Reuse in future notebooks

---

## Troubleshooting

### Problem: "GPU not available"

**Solution:**
1. Settings → Accelerator → GPU → Save
2. Runtime → Restart Session
3. Re-run Cell 1

### Problem: "Dataset not found"

**Solution:**
1. Click "Add Data" button
2. Search for your dataset name
3. Click "Add"
4. Check path: `!ls /kaggle/input/`

### Problem: "Out of memory"

**Solution:**
Edit `configs/train_kaggle.yaml`:
```yaml
training_data:
  micro_batch_size: 32  # Reduce from 64
```

### Problem: "Session expired"

**Solution:**
Checkpoints auto-saved every 1000 steps. Use resume cell:
```python
import glob
checkpoints = sorted(glob.glob('/kaggle/working/checkpoints/*.pt'))
latest = checkpoints[-1]

!python scripts/train.py \
  --config configs/train_kaggle.yaml \
  --resume_from {latest} \
  --steps 20000
```

---

## Performance Expectations

| Steps | Time | GPU Hours | Output |
|-------|------|-----------|--------|
| 100 | 30 sec | 0.01h | Smoke test |
| 1,000 | 2 min | 0.03h | Initial validation |
| 10,000 | 10-15 min | 0.2h | Short run |
| 50,000 | ~1 hour | 1h | Production run |

**Weekly Quota:** 30 GPU hours
**Capacity:** ~1.8M training steps/week

---

## Quick Reference: File Paths

```
Dataset:   /kaggle/input/graphmer-knowledge-graph/
           ├── enhanced_multilang.jsonl (30,826 triples)
           ├── src/
           ├── configs/
           ├── scripts/
           └── docs/specs/ontology_spec.yaml

Working:   /kaggle/working/
           ├── checkpoints/
           ├── outputs/
           ├── train_metrics.csv
           └── tensorboard_logs/

Config:    /kaggle/working/configs/train_kaggle.yaml
Script:    /kaggle/working/scripts/train.py
```

---

## Configuration: train_kaggle.yaml

**Key settings:**
```yaml
hardware:
  device: cuda

training_data:
  micro_batch_size: 64  # 2x local GPU
  max_seq_len: 768

run:
  mixed_precision: fp16  # Faster + less memory
  save_interval_steps: 1000

checkpointing:
  save_total_limit: 5  # File count optimization
  output_dir: /kaggle/working/checkpoints
```

---

## Next Steps

After successful training:

1. **Download outputs:** Use tarball method above
2. **Analyze metrics:** Load `train_metrics.csv` locally
3. **Iterate:** Tune hyperparameters, run ablations
4. **Scale up:** 50k-100k step production runs

---

## Summary

**Total time:** ~10 minutes
**Steps:**
1. ✅ Build dataset (2 min)
2. ✅ Upload to Kaggle (3 min)
3. ✅ Create notebook (3 min)
4. ✅ Run training (2 min setup + 10-15 min training)

**Result:** GraphMER training on Kaggle GPU (2-3x faster than local)

---

**Created:** 2025-10-21
**Platform:** Kaggle Notebooks (Tesla T4/P100)
**Status:** ✅ Ready to use

For detailed documentation, see `KAGGLE_SETUP.md`

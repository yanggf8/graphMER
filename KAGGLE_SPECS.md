# Kaggle Notebook Specifications for GraphMER Training (2025)

## Executive Summary

Kaggle notebooks provide free GPU access with generous compute quotas, making them viable for GraphMER training. Key advantages over Colab: persistent session execution (notebooks continue running after closing), better GPU availability, and integrated dataset management. Primary limitation: 30 GPU hours/week quota.

---

## 1. GPU Options & Specifications

| GPU Type | VRAM | RAM | CPU Cores | Availability |
|----------|------|-----|-----------|--------------|
| NVIDIA Tesla P100 | 16GB | 29GB | 4 cores | Standard |
| NVIDIA Tesla T4 (Single) | 16GB | 29GB | 4 cores | Standard |
| NVIDIA Tesla T4 (2x) | 32GB total | 29GB | 4 cores | Available |

**Key Points:**
- Both P100 and T4 offer 16GB VRAM (same as Colab's T4)
- Significant RAM upgrade: 29GB vs Colab's typical 12-13GB
- T4 benchmarks show ~23-25% faster performance than Colab's free T4
- 2x T4 configuration provides 32GB total VRAM for larger models

**GraphMER Compatibility:**
- Current GPU config (RTX 3050, 8GB VRAM) uses 32 batch size
- Kaggle's 16GB VRAM supports batch size 64-96 (2-3x local capacity)
- Sufficient for current model architecture (768 hidden, 12 layers)

---

## 2. Session Limits & Quotas

| Resource | Limit | Notes |
|----------|-------|-------|
| **GPU Hours/Week** | 30 hours | Shared across all GPU sessions |
| **Session Duration** | 12 hours max | Auto-terminates after 12h |
| **CPU Hours/Week** | Unlimited | 12h session limit still applies |
| **Concurrent Sessions** | Limited | Check current Kaggle policy |
| **Idle Timeout** | No automatic disconnect | Notebooks continue running when closed |

**Advantages Over Colab:**
- Notebooks run persistently (close browser, training continues)
- More predictable GPU availability
- Can "commit" notebook and check results later
- No random disconnections during active sessions

**GraphMER Training Estimates:**
- 1000 steps at 5-8 steps/sec = 2-3 minutes (local RTX 3050)
- With 16GB VRAM + 2x batch size: ~1000 steps in 1-2 minutes
- 30 hours/week = 900-1800k steps potential (sufficient for experiments)

---

## 3. Storage Limits

| Storage Type | Limit | Details |
|--------------|-------|---------|
| **Working Directory** | 20GB | `/kaggle/working/` - persists across sessions |
| **Output Files** | 20GB total | Saved to `/kaggle/working/` |
| **File Count Limit** | 500 files max | Use compression or `/tmp/` for temporary files |
| **Temporary Storage** | Additional space | `/tmp/` directory (not persistent) |
| **Input Data** | 73GB max | Combined size of added datasets |

**Workarounds:**
- Compress large outputs (checkpoints, logs) to stay under 20GB
- Use `/tmp/` for intermediate files (model caches, temporary data)
- Save only essential checkpoints (every 1000 steps vs 500)
- Stream large datasets from Kaggle datasets instead of copying

**GraphMER Storage Requirements:**
- Dataset: `enhanced_multilang.jsonl` (~50MB for 30k triples)
- Checkpoints: ~500MB per checkpoint (model + optimizer state)
- Outputs: Metrics CSV, validation results (~10MB)
- Total: <5GB for full training run (well within 20GB limit)

---

## 4. Kaggle Datasets

### Upload Limits

| Item | Limit | Details |
|------|-------|---------|
| **Dataset Size** | 20GB per dataset | Maximum size for private/public datasets |
| **File Size (Web)** | 500MB per file | Web upload interface limit |
| **File Size (API)** | 2GB per file | Using Kaggle API (`kaggle datasets`) |
| **Total Account Storage** | 20GB | Aggregate limit across all private datasets |

### Uploading Private Datasets

**Method 1: Web Interface (Recommended for Small Files)**
```bash
# 1. Prepare data
zip dataset.zip enhanced_multilang.jsonl enhanced_multilang.entities.jsonl

# 2. Upload via Kaggle website
# - Navigate to kaggle.com/datasets
# - Click "New Dataset"
# - Upload zip file
# - Set visibility to "Private"
# - Title: "graphmer-training-data"
```

**Method 2: Kaggle API (Recommended for Large Files)**
```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Download API credentials
# - Go to kaggle.com/settings
# - Scroll to "API" section
# - Click "Create New API Token"
# - Save kaggle.json to ~/.kaggle/

# 3. Create dataset metadata
cat > dataset-metadata.json << EOF
{
  "title": "graphmer-training-data",
  "id": "yourusername/graphmer-training-data",
  "licenses": [{"name": "CC0-1.0"}],
  "isPrivate": true
}
EOF

# 4. Upload dataset
kaggle datasets create -p /path/to/data/
```

### Accessing Datasets in Notebooks

**Method 1: Add Data Button (Easiest)**
```python
# 1. Click "Add Data" in notebook right sidebar
# 2. Search for your dataset: "graphmer-training-data"
# 3. Click "Add"
# 4. Dataset appears at /kaggle/input/graphmer-training-data/

# Verify in notebook
import os
print(os.listdir('/kaggle/input/graphmer-training-data/'))
# Expected: ['enhanced_multilang.jsonl', 'enhanced_multilang.entities.jsonl']
```

**Method 2: Kaggle API (Programmatic)**
```python
# Download dataset to working directory
!kaggle datasets download -d yourusername/graphmer-training-data
!unzip graphmer-training-data.zip -d /kaggle/working/data/
```

**GraphMER Dataset Strategy:**
- Upload `data/enhanced_multilang.jsonl` + entities file as single dataset (~50MB)
- Keep as private dataset (protects training data)
- Add to notebook via "Add Data" button
- Update training config to use `/kaggle/input/graphmer-training-data/`

---

## 5. Notebook Format & Features

### File Format
- **Native Format:** Jupyter Notebook (`.ipynb`)
- **Full Compatibility:** Works with Colab `.ipynb` files
- **Conversion:** No conversion needed from Colab

### Internet Access
```python
# Enable internet in notebook settings
# Settings → Internet → "Internet connected"
# Note: Requires phone verification

# Internet allows:
# - pip install packages
# - Download pretrained models (transformers)
# - Access external APIs
# - Clone git repositories

# Offline mode:
# - Required for some competitions (prevents data leakage)
# - All dependencies must be pre-installed
# - Use dataset upload for pretrained models
```

### Key Features
| Feature | Status | Notes |
|---------|--------|-------|
| **Multi-cell Execution** | Yes | Standard Jupyter interface |
| **GPU Switching** | Yes | Can switch P100 ↔ T4 in settings |
| **Version Control** | Built-in | Auto-saves versions ("commits") |
| **Markdown Cells** | Yes | Full Jupyter markdown support |
| **Magic Commands** | Yes | `%%time`, `!bash`, etc. |
| **Widgets** | Limited | Some ipywidgets supported |
| **Collaboration** | Public sharing | Notebooks can be forked/shared |

### Special Kaggle Features
```python
# 1. Persistent execution
# - Close browser → notebook keeps running
# - Check results later by reopening

# 2. Version commits
# - Click "Save Version" to create snapshot
# - Select "Save & Run All" to execute and save
# - Access past versions from "Versions" tab

# 3. Output preservation
# - Outputs saved with notebook version
# - View results without re-running

# 4. Dataset linking
# - Input datasets mounted at /kaggle/input/
# - Output files saved to /kaggle/working/
# - Outputs can be shared as datasets
```

---

## 6. Key Differences from Colab

### Advantages for ML Training

| Feature | Kaggle | Colab | Advantage |
|---------|--------|-------|-----------|
| **Session Persistence** | Continues when closed | Must keep browser open | Kaggle |
| **GPU Availability** | More consistent | Depends on demand | Kaggle |
| **RAM** | 29GB | 12-13GB | Kaggle (2.2x) |
| **Idle Disconnect** | No auto-disconnect | Disconnects after 90min idle | Kaggle |
| **Dataset Management** | Integrated datasets | Manual Drive mounting | Kaggle |
| **Version Control** | Built-in commits | Manual via Drive | Kaggle |
| **Output Preservation** | Auto-saved with version | Must save to Drive | Kaggle |
| **File Count Limit** | 500 files max | No limit | Colab |

### Limitations to Watch

| Limitation | Impact on GraphMER | Mitigation |
|------------|-------------------|------------|
| **30 GPU hours/week** | ~900-1800k steps max | Schedule long runs efficiently |
| **12h session limit** | Must checkpoint every <12h | Auto-checkpoint every 1000 steps |
| **20GB output limit** | Limits checkpoint storage | Compress checkpoints, keep last 5 only |
| **500 file limit** | Many small files problematic | Bundle logs into single CSV |
| **No Google Drive** | Can't auto-save to Drive | Download outputs or use Kaggle datasets |
| **Phone Verification** | Required for internet | Complete verification before training |

### Best Practices

**1. Session Management**
```python
# Check session time remaining
import time
start_time = time.time()

# ... training code ...

elapsed = (time.time() - start_time) / 3600
print(f"Session time used: {elapsed:.2f} hours / 12 hours max")
```

**2. Checkpoint Strategy**
```python
# Save checkpoints to /kaggle/working/ (persists)
# Keep only last 5 checkpoints to save space

import glob
checkpoints = sorted(glob.glob('/kaggle/working/checkpoint_*.pt'))
if len(checkpoints) > 5:
    for old_ckpt in checkpoints[:-5]:
        os.remove(old_ckpt)
        print(f"Removed old checkpoint: {old_ckpt}")
```

**3. Output Management**
```python
# Compress outputs before session ends
!tar -czf outputs.tar.gz /kaggle/working/outputs/
!tar -czf checkpoints.tar.gz /kaggle/working/checkpoint_*.pt

# Download compressed archives (much faster)
```

**4. GPU Quota Tracking**
```python
# Monitor weekly GPU usage
# - Check "Settings" → "Account" → "GPU Quota"
# - Plan training runs around weekly reset (Monday 00:00 UTC)
# - Use CPU sessions for debugging
```

---

## 7. GraphMER Training Adaptation

### Recommended Workflow

**Phase 1: Setup (CPU session, 0 GPU hours)**
```python
# 1. Create private dataset with training data
# 2. Upload via Kaggle website or API
# 3. Create new notebook
# 4. Add dataset via "Add Data" button
# 5. Install dependencies (CPU session)
!pip install transformers datasets pyyaml networkx
```

**Phase 2: Validation (CPU session, 0 GPU hours)**
```python
# Run data validation on CPU
!python src/ontology/kg_validator.py \
  /kaggle/input/graphmer-training-data/enhanced_multilang.jsonl \
  /kaggle/input/graphmer-training-data/enhanced_multilang.entities.jsonl \
  /kaggle/input/ontology-spec/ontology_spec.yaml

# Expected: domain_range_ratio ≥ 0.99, inherits_acyclic: True
```

**Phase 3: Training (GPU session)**
```python
# Switch to GPU: Settings → Accelerator → GPU T4 x2

# Run training with Kaggle-optimized config
!python scripts/train.py \
  --config configs/train_kaggle.yaml \
  --steps 10000 \
  --output_dir /kaggle/working/outputs \
  --checkpoint_dir /kaggle/working/checkpoints

# Expected: ~10k steps in 1-2 hours (leaves 28h GPU quota)
```

**Phase 4: Export Results**
```python
# Compress outputs
!tar -czf /kaggle/working/graphmer_outputs.tar.gz \
  /kaggle/working/outputs/ \
  /kaggle/working/checkpoints/

# Download via notebook interface or save as dataset
```

### Configuration Changes for Kaggle

**Create `configs/train_kaggle.yaml`:**
```yaml
# Key changes from train_gpu.yaml:

hardware:
  device: cuda  # Kaggle auto-detects GPU
  num_workers: 4  # Kaggle has 4 CPU cores

training_data:
  micro_batch_size: 64  # 2x local (16GB VRAM vs 8GB)

run:
  save_interval_steps: 1000  # Less frequent (500 file limit)

checkpointing:
  keep_last_n: 5  # Limit checkpoint count (20GB output limit)

# Update data paths:
# /kaggle/input/graphmer-training-data/enhanced_multilang.jsonl
# /kaggle/input/graphmer-training-data/enhanced_multilang.entities.jsonl
```

### Estimated Performance

| Metric | Local RTX 3050 | Kaggle T4 | Kaggle T4 x2 |
|--------|---------------|-----------|--------------|
| **VRAM** | 8GB | 16GB | 32GB |
| **Batch Size** | 32 | 64 | 96-128 |
| **Steps/Sec** | 5-8 | 10-15 | 15-25 |
| **1k Steps** | 2-3 min | 1-1.5 min | 40-70 sec |
| **10k Steps** | 20-30 min | 10-15 min | 7-11 min |
| **Weekly Quota** | Unlimited | 1.8M steps | 2.7M steps |

**Cost-Benefit Analysis:**
- Kaggle: Free, 30h/week, persistent sessions, 2-3x faster than local
- Colab: Free, less reliable GPU, requires open browser, similar speed
- Local: Unlimited time, slower, always available, easier debugging

**Recommendation:** Use Kaggle for production training runs (10k+ steps), local for debugging and experimentation.

---

## 8. Migration Checklist

### Pre-Migration
- [ ] Create Kaggle account and verify phone number
- [ ] Upload training data as private dataset (~50MB)
- [ ] Upload ontology spec as private dataset
- [ ] Test dataset access in CPU notebook
- [ ] Install dependencies and verify imports

### Configuration
- [ ] Create `configs/train_kaggle.yaml` with updated paths
- [ ] Update batch size to 64 (leverage 16GB VRAM)
- [ ] Set checkpoint interval to 1000 steps (file limit)
- [ ] Configure keep_last_n=5 checkpoints (storage limit)
- [ ] Update data paths to `/kaggle/input/`

### Testing
- [ ] Run 100-step test on CPU (verify setup)
- [ ] Switch to GPU and run 100-step test
- [ ] Verify checkpoint saving/loading
- [ ] Check output file sizes
- [ ] Confirm metrics logging works

### Production
- [ ] Run full 10k step training
- [ ] Monitor GPU quota usage
- [ ] Download outputs/checkpoints
- [ ] Compare results with local training
- [ ] Document any issues/optimizations

---

## 9. Troubleshooting

### Common Issues

**Problem:** "Your notebook tried to use more disk space than is available"
```python
# Solution: Clean up intermediate files
!rm -rf /tmp/*
!rm -rf /kaggle/working/cache/*

# Keep only essential checkpoints
!ls -lh /kaggle/working/checkpoint_*.pt
# Remove older checkpoints manually
```

**Problem:** "GPU quota exceeded"
```python
# Solution: Check remaining quota
# Settings → Account → GPU Quota
# Wait for Monday 00:00 UTC reset
# Use CPU for debugging meanwhile
```

**Problem:** "Session terminated after 12 hours"
```python
# Solution: Design for 12h sessions
# Auto-checkpoint every 1000 steps
# Resume from latest checkpoint:
!python scripts/train.py \
  --config configs/train_kaggle.yaml \
  --resume_from /kaggle/working/checkpoints/checkpoint_latest.pt
```

**Problem:** "Cannot access private dataset"
```python
# Solution: Ensure dataset is added to notebook
# Click "Add Data" → Search → Select dataset → Add
# Verify path: /kaggle/input/<dataset-slug>/
```

### Performance Optimization

**1. Maximize Throughput**
```yaml
# configs/train_kaggle.yaml
training_data:
  micro_batch_size: 64  # Use full 16GB VRAM
  num_workers: 4  # All CPU cores
  pin_memory: true  # Faster GPU transfer
  prefetch_factor: 2  # Prefetch batches
```

**2. Minimize Checkpointing Overhead**
```python
# Save checkpoints asynchronously
# Only save model state (not optimizer) for intermediate checkpoints
# Full checkpoint every 5000 steps, model-only every 1000 steps
```

**3. Efficient Data Loading**
```python
# Use memory-mapped files for large datasets
import mmap
# Or stream from dataset without copying to /kaggle/working/
```

---

## 10. Comparison Table

| Feature | Kaggle | Colab | Local RTX 3050 |
|---------|--------|-------|----------------|
| **GPU VRAM** | 16-32GB | 16GB | 8GB |
| **System RAM** | 29GB | 12-13GB | 16GB |
| **CPU Cores** | 4 | 2 | 8 |
| **Session Limit** | 12h | 12h | Unlimited |
| **Weekly Quota** | 30h GPU | Variable | Unlimited |
| **Persistent Execution** | Yes | No | Yes |
| **Storage Limit** | 20GB | 100GB Drive | 1TB SSD |
| **File Count Limit** | 500 | Unlimited | Unlimited |
| **Internet Access** | Yes (w/ verify) | Yes | Yes |
| **Cost** | Free | Free | Hardware cost |
| **Batch Size** | 64-128 | 64 | 32 |
| **Steps/Sec** | 10-25 | 8-15 | 5-8 |
| **Best For** | Production runs | Quick experiments | Development |

---

## 11. Quick Reference

### Essential Paths
```bash
# Input datasets (read-only)
/kaggle/input/<dataset-name>/

# Working directory (read-write, 20GB limit, persists)
/kaggle/working/

# Temporary storage (read-write, not persistent)
/tmp/

# Example paths for GraphMER
/kaggle/input/graphmer-training-data/enhanced_multilang.jsonl
/kaggle/working/outputs/train_metrics.csv
/kaggle/working/checkpoints/checkpoint_10000.pt
```

### Essential Commands
```python
# Check GPU
!nvidia-smi

# Check disk usage
!df -h /kaggle/working/
!du -sh /kaggle/working/*

# Check file count
!find /kaggle/working/ -type f | wc -l

# Check session time
import time
start = time.time()
# ... later ...
elapsed_hours = (time.time() - start) / 3600
print(f"{elapsed_hours:.2f}h / 12h used")

# Monitor training
!tail -f /kaggle/working/outputs/train.log
```

### Key Settings
```python
# Enable GPU: Settings → Accelerator → GPU T4 x2
# Enable Internet: Settings → Internet → Internet connected
# Save version: File → Save Version → Save & Run All
# Check quota: Settings → Account → GPU Quota
```

---

## 12. Conclusion

**Kaggle is highly suitable for GraphMER training:**
- 16-32GB VRAM supports 2-4x larger batches than local GPU
- 30 GPU hours/week = sufficient for weekly experimentation (1.8M+ steps)
- Persistent execution eliminates browser babysitting
- Integrated dataset management simplifies data workflows
- No cost vs cloud GPU providers

**Primary consideration:** Weekly quota management. Plan production runs around Monday resets, use CPU sessions for debugging, and optimize training efficiency to maximize experiments per week.

**Next steps:** Create Kaggle account, upload training data, run 100-step test, then execute full training run.

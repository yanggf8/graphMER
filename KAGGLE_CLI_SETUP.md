# Kaggle CLI Setup Guide

**Status:** ✅ Kaggle CLI installed (version 1.7.4.5)

---

## Setup Steps

### Step 1: Get API Credentials

1. **Go to your Kaggle account settings:**
   ```
   https://www.kaggle.com/YOUR_USERNAME/account
   ```

2. **Scroll down to "API" section**

3. **Click: "Create New API Token"**
   - This downloads a file: `kaggle.json`
   - Location: Usually `~/Downloads/kaggle.json`

4. **Move credentials to correct location:**
   ```bash
   # Create Kaggle config directory
   mkdir -p ~/.kaggle

   # Move the downloaded file
   mv ~/Downloads/kaggle.json ~/.kaggle/

   # Set correct permissions (required for security)
   chmod 600 ~/.kaggle/kaggle.json
   ```

5. **Verify setup:**
   ```bash
   kaggle datasets list --mine
   ```
   - Should list your datasets (empty if you haven't uploaded any yet)

---

## Upload Dataset Using CLI

### Method 1: Quick Upload (Simple)

```bash
# Build the dataset package first
cd /home/yanggf/a/graphMER
./scripts/deploy_to_kaggle.sh

# Navigate to deploy directory
cd kaggle_deploy

# IMPORTANT: Edit metadata file first!
nano dataset-metadata.json
# Change line 3: "id": "YOUR_USERNAME/graphmer-kg"
# Replace YOUR_USERNAME with your actual Kaggle username

# Create the dataset
kaggle datasets create -p .

# Expected output:
# Starting upload for file ...
# Upload successful: ...
# Dataset URL: https://www.kaggle.com/datasets/YOUR_USERNAME/graphmer-kg
```

### Method 2: Update Existing Dataset

If you already created the dataset via web and want to update it:

```bash
cd /home/yanggf/a/graphMER/kaggle_deploy

# Version the dataset with a message
kaggle datasets version -p . -m "Updated to 30,826 triples with improved quality"
```

---

## Common CLI Commands

### List Your Datasets
```bash
kaggle datasets list --mine
```

### Download a Dataset
```bash
kaggle datasets download -d YOUR_USERNAME/graphmer-kg
```

### View Dataset Info
```bash
kaggle datasets status YOUR_USERNAME/graphmer-kg
```

### List Your Notebooks
```bash
kaggle kernels list --mine
```

---

## Quick Setup Script

Here's a one-liner to set everything up (after you download kaggle.json):

```bash
mkdir -p ~/.kaggle && \
mv ~/Downloads/kaggle.json ~/.kaggle/ && \
chmod 600 ~/.kaggle/kaggle.json && \
kaggle datasets list --mine && \
echo "✅ Kaggle CLI configured successfully!"
```

---

## Troubleshooting

### Error: "Unauthorized: Invalid credentials"

**Cause:** API token not found or incorrect permissions

**Solution:**
```bash
# Check if file exists
ls -la ~/.kaggle/kaggle.json

# Check permissions (should be -rw-------)
# If not, fix it:
chmod 600 ~/.kaggle/kaggle.json

# Verify file content (should be valid JSON)
cat ~/.kaggle/kaggle.json
# Should show: {"username":"...","key":"..."}
```

### Error: "Could not find kaggle.json"

**Cause:** File not in correct location

**Solution:**
```bash
# Search for the file
find ~ -name "kaggle.json" 2>/dev/null

# Move to correct location
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Error: "403 Forbidden"

**Cause:** Token expired or revoked

**Solution:**
1. Go to kaggle.com/account
2. Click "Expire API Token"
3. Click "Create New API Token"
4. Download new `kaggle.json`
5. Replace old file: `mv ~/Downloads/kaggle.json ~/.kaggle/`

---

## Comparison: Web Upload vs CLI Upload

| Feature | Web Upload | CLI Upload |
|---------|------------|------------|
| **Setup Time** | 0 min | 5 min (one-time) |
| **Max File Size** | 500 MB | 2 GB per file |
| **Convenience** | Drag & drop | Command line |
| **Automation** | ❌ No | ✅ Yes |
| **Versioning** | Manual | Automated |
| **Best For** | First upload | Updates, large files |

---

## Recommendation for GraphMER

**Your dataset size:** ~3-5 MB
**Recommendation:** Either method works fine!

### Use Web Upload if:
- ✅ First time uploading
- ✅ You prefer GUI
- ✅ One-time setup

### Use CLI Upload if:
- ✅ You'll update the dataset frequently
- ✅ You want to automate deployment
- ✅ You're comfortable with command line

---

## Complete CLI Workflow

```bash
# 1. Setup (one-time)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Build dataset package
cd /home/yanggf/a/graphMER
./scripts/deploy_to_kaggle.sh

# 3. Edit metadata
cd kaggle_deploy
nano dataset-metadata.json
# Update: "id": "YOUR_USERNAME/graphmer-kg"

# 4. Upload
kaggle datasets create -p .

# 5. Verify
kaggle datasets list --mine

# Done! Dataset is now available at:
# https://www.kaggle.com/datasets/YOUR_USERNAME/graphmer-kg
```

---

## Next Steps After Upload

**Via CLI:**
```bash
# Check dataset status
kaggle datasets status YOUR_USERNAME/graphmer-kg

# Download to verify (optional)
kaggle datasets download -d YOUR_USERNAME/graphmer-kg -p /tmp/test
ls -lh /tmp/test/
```

**Via Web:**
1. Go to: https://www.kaggle.com/datasets/YOUR_USERNAME/graphmer-kg
2. Verify files are present
3. Check dataset visibility (Private/Public)
4. Copy dataset ID for notebook

**In Notebook:**
1. Create new notebook: https://www.kaggle.com/code
2. Settings → GPU → On
3. Click "Add Data" → Search "graphmer-kg"
4. Dataset mounts at: `/kaggle/input/graphmer-kg/`

---

## API Token Security

**IMPORTANT:** Keep your `kaggle.json` secure!

**DO:**
- ✅ Keep permissions at 600 (`chmod 600`)
- ✅ Store only in `~/.kaggle/`
- ✅ Add to `.gitignore` if in project directory
- ✅ Rotate token periodically

**DON'T:**
- ❌ Commit to git repositories
- ❌ Share publicly
- ❌ Store in cloud sync folders (Dropbox, Drive)
- ❌ Set world-readable permissions

---

## Summary

**Kaggle CLI is now installed!** ✅

**Two options for uploading your dataset:**

1. **Web Upload (Easiest):**
   - Run: `./scripts/deploy_to_kaggle.sh`
   - Go to: https://www.kaggle.com/datasets
   - Upload: `graphmer_kaggle_dataset.zip`
   - Done!

2. **CLI Upload (More powerful):**
   - Get API token from kaggle.com/account
   - Setup: `~/.kaggle/kaggle.json`
   - Edit: `kaggle_deploy/dataset-metadata.json`
   - Upload: `kaggle datasets create -p kaggle_deploy/`

**Both methods work perfectly for your 3-5 MB dataset.**

---

**Created:** 2025-10-21
**Kaggle CLI Version:** 1.7.4.5
**Status:** Ready to use

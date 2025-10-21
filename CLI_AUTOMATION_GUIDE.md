# GraphMER-SE: CLI Automation Guide for Google Drive & Colab TPU

**Date:** October 20, 2025  
**Purpose:** Automate package upload to Google Drive using CLI tools  
**Recommended Tool:** **rclone** (cross-platform, powerful, free)

---

## Why Use CLI Automation?

✅ **No manual browser uploads**  
✅ **Scriptable and repeatable**  
✅ **Resume failed uploads**  
✅ **Integrate into CI/CD pipelines**  
✅ **Batch uploads for multiple experiments**  

---

## Recommended: rclone

**Best for:** Most users, cross-platform, actively maintained

### Quick Start (3 Steps)

#### Step 1: Install rclone

```bash
# Automated installation
./scripts/setup_rclone.sh

# OR manual installation
curl https://rclone.org/install.sh | sudo bash

# Verify installation
rclone version
```

#### Step 2: Configure Google Drive

```bash
# Start configuration wizard
rclone config

# Follow these prompts:
# n) New remote
# name> gdrive
# Storage> drive (Google Drive - option 15 or similar)
# client_id> (press Enter - leave blank)
# client_secret> (press Enter - leave blank)
# scope> 1 (Full access)
# root_folder_id> (press Enter)
# service_account_file> (press Enter)
# Edit advanced config? n
# Use auto config? y (opens browser for authentication)
# Configure this as a Shared Drive? n
# Keep this remote? y
```

**Authentication:** Opens browser → Log in to Google → Grant permissions

#### Step 3: Upload Package

```bash
# Automated upload (recommended)
./scripts/upload_to_drive.sh

# OR manual upload
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ --progress
```

**Done!** File is now at: `Google Drive > GraphMER > graphmer_colab.tar.gz`

---

## End-to-End Automation

### One-Command Deployment

```bash
# Validates KG → Creates package → Uploads to Drive
./scripts/deploy_to_colab.sh
```

**This script:**
1. ✅ Validates knowledge graph (30k+ triples, 99%+ quality)
2. ✅ Creates `colab_deploy/` with all necessary files
3. ✅ Generates `graphmer_colab.tar.gz` tarball
4. ✅ Uploads to Google Drive (if rclone configured)
5. ✅ Prints Colab notebook setup instructions

---

## CLI Tools Comparison

| Tool | Pros | Cons | Best For |
|------|------|------|----------|
| **rclone** ⭐ | Cross-platform, powerful, resumable uploads, well-documented | Requires OAuth setup | Most users, production |
| **gdrive** | Simple, single binary, fast for one-offs | Less feature-rich | Quick uploads |
| **drive-cli** | Python-based, pip install | Requires Python | Python users |
| **gcloud** | Official Google tool | Complex setup, requires GCP project | Enterprise/GCS users |

**Recommendation:** Use **rclone** - it's the most versatile and reliable.

---

## rclone: Detailed Commands

### Basic Operations

```bash
# List remotes
rclone listremotes

# List folders in Google Drive root
rclone lsd gdrive:

# List files in GraphMER folder
rclone ls gdrive:GraphMER/

# Create folder
rclone mkdir gdrive:GraphMER/

# Upload single file
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ --progress

# Upload entire directory
rclone copy colab_deploy/ gdrive:GraphMER/colab_deploy/ --progress

# Download from Drive
rclone copy gdrive:GraphMER/graphmer_colab.tar.gz ./downloads/

# Delete file
rclone delete gdrive:GraphMER/old_package.tar.gz

# Sync directory (two-way sync)
rclone sync colab_deploy/ gdrive:GraphMER/colab_deploy/
```

### Advanced Options

```bash
# Upload with bandwidth limit (1 MB/s)
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ --bwlimit 1M

# Upload with retries (5 attempts)
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ --retries 5

# Upload with checksum verification
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ --checksum

# Dry run (preview without uploading)
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ --dry-run

# Upload with detailed logging
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ -v --log-file upload.log
```

---

## Alternative CLI Tools

### Option 2: gdrive (Simple Go Binary)

**Installation:**
```bash
# Download latest release
wget https://github.com/prasmussen/gdrive/releases/download/2.1.1/gdrive_linux-x64
chmod +x gdrive_linux-x64
sudo mv gdrive_linux-x64 /usr/local/bin/gdrive

# Authenticate
gdrive about
# Opens browser for Google login
```

**Usage:**
```bash
# Upload file
gdrive upload graphmer_colab.tar.gz

# Upload to specific folder (get folder ID via 'gdrive list')
gdrive upload --parent FOLDER_ID graphmer_colab.tar.gz

# List files
gdrive list

# Download file
gdrive download FILE_ID
```

**Pros:** Very simple, single binary  
**Cons:** Less feature-rich than rclone

---

### Option 3: drive-cli (Python Tool)

**Installation:**
```bash
pip install drive-cli
```

**Setup:**
```bash
# Authenticate
drive auth
# Opens browser for Google login
```

**Usage:**
```bash
# Upload file
drive upload graphmer_colab.tar.gz

# Upload to folder
drive upload graphmer_colab.tar.gz --parent FOLDER_NAME

# List files
drive ls

# Download file
drive download FILENAME
```

**Pros:** Python-native, easy pip install  
**Cons:** Requires Python, less performant than rclone

---

## Automation Scripts Created

### 1. `scripts/setup_rclone.sh`
Installs and provides configuration instructions for rclone.

```bash
./scripts/setup_rclone.sh
```

### 2. `scripts/upload_to_drive.sh`
Automated upload with validation and progress display.

```bash
./scripts/upload_to_drive.sh
```

**Features:**
- ✅ Checks rclone installation
- ✅ Verifies package exists
- ✅ Creates Drive folder if needed
- ✅ Shows upload progress
- ✅ Verifies upload success
- ✅ Provides next steps

### 3. `scripts/deploy_to_colab.sh`
End-to-end deployment pipeline.

```bash
./scripts/deploy_to_colab.sh
```

**Pipeline:**
1. Validates knowledge graph (30k+ triples, 99%+ quality)
2. Creates deployment package
3. Generates Colab-optimized config
4. Creates tarball
5. Uploads to Google Drive (if rclone configured)
6. Prints Colab setup instructions

---

## Troubleshooting

### Issue: "rclone: command not found"

**Solution:**
```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# OR use package manager
sudo apt install rclone  # Debian/Ubuntu
brew install rclone      # macOS
```

### Issue: "Failed to configure token: invalid_grant"

**Solution:**
Reauthenticate:
```bash
rclone config reconnect gdrive:
```

### Issue: "Upload stuck or very slow"

**Solution:**
Use bandwidth control and retries:
```bash
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ \
  --bwlimit 10M \
  --retries 10 \
  --low-level-retries 10
```

### Issue: "OAuth token expired"

**Solution:**
Refresh token:
```bash
rclone config reconnect gdrive:
```

Or reconfigure:
```bash
rclone config
# Select existing 'gdrive' remote
# Choose 'reconnect'
```

### Issue: "File exists but Colab can't find it"

**Solution:**
Check path in Colab:
```python
# In Colab notebook
!ls -lh /content/drive/MyDrive/GraphMER/
```

Ensure file uploaded to correct folder:
```bash
rclone ls gdrive:GraphMER/
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Deploy to Colab

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install rclone
        run: curl https://rclone.org/install.sh | sudo bash
      
      - name: Configure rclone
        env:
          RCLONE_CONFIG: ${{ secrets.RCLONE_CONFIG }}
        run: |
          mkdir -p ~/.config/rclone
          echo "$RCLONE_CONFIG" > ~/.config/rclone/rclone.conf
      
      - name: Build and deploy
        run: ./scripts/deploy_to_colab.sh
```

Store rclone config in GitHub Secrets:
```bash
# Get config
cat ~/.config/rclone/rclone.conf

# Add to GitHub: Settings → Secrets → RCLONE_CONFIG
```

---

## Performance Tips

### 1. Parallel Uploads (for multiple files)

```bash
# Upload multiple packages in parallel
rclone copy experiments/ gdrive:GraphMER/experiments/ \
  --transfers 4 \
  --checkers 8
```

### 2. Compression Before Upload

```bash
# Create smaller package (better compression)
tar -czf graphmer_colab.tar.gz colab_deploy/ --best

# Verify size
ls -lh graphmer_colab.tar.gz
```

### 3. Resume Interrupted Uploads

rclone automatically resumes interrupted uploads. No special flag needed!

### 4. Cache Directory Listings

```bash
# Speed up repeated operations
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ \
  --cache-dir ~/.cache/rclone
```

---

## Security Best Practices

### 1. Limit Scope

When configuring rclone, choose the minimal scope needed:
- **Full access (1):** Read/write/delete all files
- **Read-only (2):** Can only read files
- **File access (3):** Can only access files created by rclone

For deployment, use **Full access (1)**.

### 2. Protect Config File

```bash
# Set restrictive permissions
chmod 600 ~/.config/rclone/rclone.conf

# View current permissions
ls -la ~/.config/rclone/rclone.conf
```

### 3. Use Service Account (Advanced)

For automated/headless environments:
```bash
# Create service account in Google Cloud Console
# Download JSON key
# Configure rclone with service account
rclone config
# Choose 'service_account_file' and provide path to JSON
```

---

## Quick Reference

### Common Commands

```bash
# Full automated deployment
./scripts/deploy_to_colab.sh

# Just upload existing package
./scripts/upload_to_drive.sh

# Manual upload
rclone copy graphmer_colab.tar.gz gdrive:GraphMER/ --progress

# List files in Drive
rclone ls gdrive:GraphMER/

# Download from Drive
rclone copy gdrive:GraphMER/graphmer_colab.tar.gz ./

# Delete file from Drive
rclone delete gdrive:GraphMER/old_file.tar.gz

# Sync local to Drive
rclone sync colab_deploy/ gdrive:GraphMER/colab_deploy/
```

### File Locations

| Location | Path |
|----------|------|
| **Local package** | `/home/yanggf/a/graphMER/graphmer_colab.tar.gz` |
| **Google Drive** | `My Drive/GraphMER/graphmer_colab.tar.gz` |
| **Colab (after extract)** | `/content/colab_deploy/` |
| **rclone config** | `~/.config/rclone/rclone.conf` |

---

## Summary

✅ **Best CLI tool:** rclone (cross-platform, powerful, reliable)  
✅ **Installation:** `./scripts/setup_rclone.sh`  
✅ **Upload:** `./scripts/upload_to_drive.sh`  
✅ **Full automation:** `./scripts/deploy_to_colab.sh`  

**Next Steps:**
1. Install rclone: `./scripts/setup_rclone.sh`
2. Configure: `rclone config` (follow prompts)
3. Deploy: `./scripts/deploy_to_colab.sh`
4. Open Colab: https://colab.research.google.com
5. Follow: `UPLOAD_INSTRUCTIONS.md`

**You now have fully automated CLI deployment to Google Drive! 🚀**

---

**Created:** 2025-10-20  
**Tool:** rclone (recommended)  
**Scripts:** setup_rclone.sh, upload_to_drive.sh, deploy_to_colab.sh  
**Status:** Production-ready automation

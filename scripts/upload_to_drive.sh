#!/bin/bash
# Automated upload to Google Drive using rclone

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}║     GraphMER-SE: Upload to Google Drive (rclone)        ║${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Configuration
PACKAGE_FILE="graphmer_colab.tar.gz"
DRIVE_REMOTE="gdrive"
DRIVE_FOLDER="GraphMER"

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo -e "${RED}❌ rclone not found${NC}"
    echo ""
    echo "Install rclone first:"
    echo "  ./scripts/setup_rclone.sh"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ rclone installed: $(rclone version | head -1)${NC}"
echo ""

# Check if rclone is configured
if ! rclone listremotes | grep -q "^${DRIVE_REMOTE}:"; then
    echo -e "${YELLOW}⚠️  rclone not configured for Google Drive${NC}"
    echo ""
    echo "Configure rclone first:"
    echo "  rclone config"
    echo ""
    echo "Follow the prompts to add a 'gdrive' remote."
    echo "See: scripts/setup_rclone.sh for detailed instructions"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ rclone configured with remote: ${DRIVE_REMOTE}${NC}"
echo ""

# Check if package exists
if [ ! -f "$PACKAGE_FILE" ]; then
    echo -e "${RED}❌ Package not found: $PACKAGE_FILE${NC}"
    echo ""
    echo "Create package first:"
    echo "  tar -czf graphmer_colab.tar.gz colab_deploy/"
    echo ""
    exit 1
fi

PACKAGE_SIZE=$(du -h "$PACKAGE_FILE" | cut -f1)
echo -e "${GREEN}✅ Package found: $PACKAGE_FILE ($PACKAGE_SIZE)${NC}"
echo ""

# Create folder on Google Drive if it doesn't exist
echo -e "${YELLOW}📁 Creating folder on Google Drive: $DRIVE_FOLDER${NC}"
rclone mkdir ${DRIVE_REMOTE}:${DRIVE_FOLDER} 2>/dev/null || true
echo ""

# Upload package
echo -e "${YELLOW}📤 Uploading $PACKAGE_FILE to Google Drive...${NC}"
echo ""

rclone copy "$PACKAGE_FILE" ${DRIVE_REMOTE}:${DRIVE_FOLDER}/ \
    --progress \
    --stats 1s \
    --stats-one-line

echo ""
echo -e "${GREEN}✅ Upload complete!${NC}"
echo ""

# Verify upload
echo -e "${YELLOW}🔍 Verifying upload...${NC}"
if rclone ls ${DRIVE_REMOTE}:${DRIVE_FOLDER}/ | grep -q "$PACKAGE_FILE"; then
    echo -e "${GREEN}✅ File verified on Google Drive${NC}"
    
    # Get file info
    rclone ls ${DRIVE_REMOTE}:${DRIVE_FOLDER}/ | grep "$PACKAGE_FILE"
else
    echo -e "${RED}❌ Upload verification failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}║                 UPLOAD SUCCESSFUL! 🚀                    ║${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "📁 Location: Google Drive > $DRIVE_FOLDER > $PACKAGE_FILE"
echo "📊 Size: $PACKAGE_SIZE"
echo ""
echo "Next steps:"
echo "1. Go to: https://colab.research.google.com"
echo "2. Create new notebook with TPU runtime"
echo "3. Follow instructions in: UPLOAD_INSTRUCTIONS.md"
echo ""
echo "Quick access to your file:"
echo "  https://drive.google.com/drive/folders/"
echo ""

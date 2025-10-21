#!/bin/bash
# Setup rclone for Google Drive automation

set -e

echo "=== Installing rclone ==="

# Install rclone
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
else
    echo "✅ rclone already installed: $(rclone version | head -1)"
fi

echo ""
echo "=== Configure rclone for Google Drive ==="
echo ""
echo "Next steps:"
echo "1. Run: rclone config"
echo "2. Choose 'n' for new remote"
echo "3. Name it: gdrive"
echo "4. Choose type: drive (Google Drive)"
echo "5. Leave client_id and client_secret blank (press Enter)"
echo "6. Choose scope: 1 (Full access)"
echo "7. Leave root_folder_id blank"
echo "8. Leave service_account_file blank"
echo "9. Choose 'n' for advanced config"
echo "10. Choose 'y' for auto config (will open browser)"
echo "11. Authenticate with your Google account"
echo "12. Choose 'y' to confirm"
echo ""
echo "After configuration, test with:"
echo "  rclone lsd gdrive:"
echo ""
echo "Upload package with:"
echo "  rclone copy graphmer_colab.tar.gz gdrive:GraphMER/"

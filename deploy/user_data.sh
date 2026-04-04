#!/bin/bash
# EC2 User Data Script — runs automatically at instance launch
# Use this as User Data when launching an EC2 instance
#
# Logs are written to /var/log/cloud-init-output.log

set -euo pipefail

REPO_URL="https://github.com/sziegler11/rf-detr-playground.git"
INSTALL_DIR="/home/ubuntu/rf-detr-playground"

# Wait for cloud-init to finish
cloud-init status --wait || true

# Run setup as ubuntu user
sudo -u ubuntu bash << 'SETUP'
cd $HOME

# Clone repo
if [ ! -d rf-detr-playground ]; then
    git clone https://github.com/sziegler11/rf-detr-playground.git
fi

cd rf-detr-playground

# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"

# Download datasets
python scripts/download_datasets.py --all

echo "RF-DETR Playground setup complete!" > /tmp/setup_complete.txt
SETUP

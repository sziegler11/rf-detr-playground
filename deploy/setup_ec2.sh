#!/bin/bash
# EC2 Setup Script for RF-DETR Playground
# Tested on: Ubuntu 22.04 LTS (Deep Learning AMI recommended)
#
# Usage:
#   chmod +x deploy/setup_ec2.sh
#   ./deploy/setup_ec2.sh

set -euo pipefail

echo "========================================="
echo "RF-DETR Playground — EC2 Setup"
echo "========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# Install Python 3.11+ if not present
echo "[2/6] Checking Python installation..."
if ! command -v python3.11 &> /dev/null; then
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

PYTHON=$(command -v python3.11 || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version))"

# Install system dependencies
echo "[3/6] Installing system dependencies..."
sudo apt-get install -y -qq \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    tmux \
    htop \
    nvtop 2>/dev/null || true

# Check for NVIDIA GPU
echo "[4/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No NVIDIA GPU detected."
    echo "For GPU support, use a Deep Learning AMI or install CUDA manually."
    echo "Recommended instances: g4dn.xlarge (budget), g5.xlarge (performance)"
fi

# Clone repo and setup
echo "[5/6] Setting up RF-DETR Playground..."
REPO_DIR="$HOME/rf-detr-playground"

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/sziegler11/rf-detr-playground.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# Create virtual environment
$PYTHON -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "[6/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[all]"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Quick start:"
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo ""
echo "  # Download datasets"
echo "  make download-data"
echo ""
echo "  # Train a model"
echo "  make train CONFIG=configs/aquarium.yaml"
echo ""
echo "  # Evaluate"
echo "  make evaluate CHECKPOINT=outputs/latest/best.pt DATASET=aquarium"
echo ""
echo "Tip: Use tmux for long training runs:"
echo "  tmux new -s training"
echo "  make train CONFIG=configs/pascal_voc.yaml"
echo "  # Ctrl+B, D to detach"

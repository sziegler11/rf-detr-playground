# EC2 Deployment Guide

## Quick Start

### 1. Launch an EC2 Instance

**Recommended Configurations:**

| Use Case | Instance Type | GPU | VRAM | Cost (approx) |
|----------|--------------|-----|------|----------------|
| Budget / experimentation | g4dn.xlarge | T4 | 16GB | ~$0.53/hr |
| Faster training | g5.xlarge | A10G | 24GB | ~$1.01/hr |
| Large datasets | g5.2xlarge | A10G | 24GB | ~$1.21/hr |

**AMI:** Use the AWS Deep Learning AMI (Ubuntu) — it comes with NVIDIA drivers, CUDA, and cuDNN pre-installed.

**Storage:** At least 50GB EBS (100GB recommended for Pascal VOC + COCO).

**Security Group:** Allow SSH (port 22) from your IP.

### 2. Automated Setup

**Option A: User Data (fully automatic)**

When launching, paste the contents of `user_data.sh` into the "User Data" field. The instance will set itself up on first boot.

**Option B: Manual SSH setup**

```bash
ssh -i your-key.pem ubuntu@<instance-ip>
git clone https://github.com/sziegler11/rf-detr-playground.git
cd rf-detr-playground
chmod +x deploy/setup_ec2.sh
./deploy/setup_ec2.sh
```

### 3. Run Training

```bash
cd ~/rf-detr-playground
source .venv/bin/activate

# Download datasets
make download-data

# Train (use tmux for long runs)
tmux new -s training
make train CONFIG=configs/aquarium.yaml

# Detach: Ctrl+B, D
# Reattach: tmux attach -t training
```

### 4. Retrieve Results

```bash
# From your local machine:
scp -i your-key.pem -r ubuntu@<ip>:~/rf-detr-playground/outputs/ ./outputs/
```

## Roboflow API Key

The Aquarium dataset requires a Roboflow API key. Get one free at https://app.roboflow.com/settings/api

```bash
export ROBOFLOW_API_KEY=your_key_here
```

## GPU Memory Guide

| Model Size | Batch Size | Grad Accum | Effective Batch | VRAM Needed |
|-----------|-----------|------------|-----------------|-------------|
| Nano      | 16        | 1          | 16              | ~8GB        |
| Base      | 4         | 4          | 16              | ~12GB       |
| Base      | 8         | 2          | 16              | ~16GB       |
| Large     | 4         | 4          | 16              | ~16GB       |
| Large     | 2         | 8          | 16              | ~12GB       |

Adjust `batch_size` and `grad_accum_steps` in your config YAML to fit your GPU.

## Monitoring

```bash
# GPU utilization
watch nvidia-smi

# If nvtop is installed
nvtop

# Training logs (if TensorBoard enabled)
tensorboard --logdir outputs/ --host 0.0.0.0 --port 6006
# Then tunnel: ssh -L 6006:localhost:6006 ubuntu@<ip>
```

## Cost Tips

- Use **Spot Instances** for 60-70% savings (training can be interrupted)
- Stop the instance when not training — you only pay for EBS storage
- Use `g4dn.xlarge` for small datasets like Aquarium (training takes ~15-30 min)
- Use `g5.xlarge` for Pascal VOC or larger datasets

#!/bin/bash
set -e

# ======================== Configuration ========================
CONFIG="configs/FourierPET_3_2.yml"
GPU_IDS="1"
# ===============================================================

# Uncomment to use real clinical data config instead:
# CONFIG="configs/FourierPET_real_clinical.yml"

export CUDA_VISIBLE_DEVICES=${GPU_IDS}

echo "============================================"
echo "  FourierPET Training"
echo "  Config : ${CONFIG}"
echo "  GPU(s) : ${GPU_IDS}"
echo "============================================"

python train.py --config_exp ${CONFIG}

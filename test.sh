#!/bin/bash
set -e

# ======================== Configuration ========================
CONFIG="configs/FourierPET_3_2.yml"
GPU_IDS="0"
DEVICE="cuda:0"
SAVE_IMAGES=false
# ===============================================================

# Uncomment to use real clinical data config instead:
# CONFIG="configs/FourierPET_real_clinical.yml"

export CUDA_VISIBLE_DEVICES=${GPU_IDS}

echo "============================================"
echo "  FourierPET Evaluation"
echo "  Config : ${CONFIG}"
echo "  GPU(s) : ${GPU_IDS}"
echo "  Device : ${DEVICE}"
echo "============================================"

CMD="python test.py --config_exp ${CONFIG} --device ${DEVICE}"

# ---------- LOOCV evaluation (simulated data) ----------
# Evaluates all LOOCV folds; checkpoints are auto-discovered
# under outputs/<exp_name>/LOOCV_k/
CMD="${CMD} --loocv"

# To evaluate a single fold only, uncomment and set fold index:
# CMD="${CMD} --loocv_fold 0"

# ---------- Single checkpoint evaluation (real clinical) ----------
# Uncomment and specify the checkpoint path for non-LOOCV configs:
# CMD="python test.py --config_exp ${CONFIG} --device ${DEVICE}"
# CMD="${CMD} --checkpoint outputs/FourierPET_real_clinical/checkpoint.pth.tar"

# ---------- Optional flags ----------
if [ "${SAVE_IMAGES}" = true ]; then
    CMD="${CMD} --save_images"
fi

echo "Running: ${CMD}"
eval ${CMD}

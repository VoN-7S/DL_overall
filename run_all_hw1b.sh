#!/usr/bin/env bash
# run_all_hw1b.sh
# ---------------
# Run all HW1b experiments through main.py in sequence.
# Edit the variables below to match your hardware and time budget.
#
# Usage:
#   chmod +x run_all_hw1b.sh
#   ./run_all_hw1b.sh
#
# Note: On Windows set NUM_WORKERS=0 to avoid DataLoader multiprocessing issues.
 
set -e  # stop immediately if any command fails
 
# ── Settings ───────────────────────────────────────────────────────────────────
EPOCHS=10
BATCH=128
LR=1e-3
NUM_WORKERS=2
SEED=7
TEMPERATURE=4.0
ALPHA=0.7
SMOOTHING=0.1
 
# ── Part A: Transfer Learning ──────────────────────────────────────────────────
echo "=============================================="
echo " HW1b Part A: Transfer Learning"
echo "=============================================="
 
python main.py \
    --task transfer \
    --tl_option both \
    --epoch $EPOCHS \
    --batch_size $BATCH \
    --learning_rate $LR \
    --num_workers $NUM_WORKERS \
    --seed $SEED
 
# ── Part B: Knowledge Distillation ────────────────────────────────────────────
echo ""
echo "=============================================="
echo " HW1b Part B: Knowledge Distillation"
echo "=============================================="
 
python main.py \
    --task distillation \
    --epoch $EPOCHS \
    --batch_size $BATCH \
    --learning_rate $LR \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --kd_temperature $TEMPERATURE \
    --kd_alpha $ALPHA \
    --kd_smoothing $SMOOTHING
 
echo ""
echo "=============================================="
echo " All done!"
echo " Results saved to:"
echo "   ./results/transfer/transfer_resize/"
echo "   ./results/transfer/transfer_layerchange/"
echo "   ./results/kd/kd_simplecnn_scratch/"
echo "   ./results/kd/kd_resnet/"
echo "   ./results/kd/kd_simplecnn_kd/"
echo "   ./results/kd/kd_mobilenet/"
echo "=============================================="

#!/bin/bash
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
set -euo pipefail

CONDA_PYTHON="/root/miniconda3/envs/calam-unsloth/bin/python3"
DATA="/jhdx0003008/data/calam/checkpoints/risk_train_data.pt"
CKPT_DIR="/jhdx0003008/data/calam/checkpoints"
LOG_DIR="logs"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

export PYTHONUNBUFFERED=1

echo "══════════════════════════════════════════════"
echo "🔍 RiskModel Parallel Sweep (5 GPUs)"
echo "   Data: $DATA"
echo "   Configs: baseline, heavy_reg, slow_learn, big_patience, high_threshold"
echo "══════════════════════════════════════════════"

COMMON="--data $DATA --epochs 50 --warmup_epochs 3 --use_wandb"

echo "🚀 [GPU 0] baseline: lr=1e-4, drop=0.3, wd=1e-4, thr=0.3"
$CONDA_PYTHON scripts/train_risk_model.py $COMMON \
    --gpu 0 --run_name "sweep_baseline" \
    --lr 1e-4 --dropout 0.3 --weight_decay 1e-4 --patience 10 --toxic_threshold 0.3 \
    --output "$CKPT_DIR/sweep_baseline_w_risk_v2.pt" \
    --risk_output "$CKPT_DIR/sweep_baseline_risk_model.pt" \
    2>&1 | tee "$LOG_DIR/sweep_baseline.log" &

echo "🚀 [GPU 1] heavy_reg: lr=1e-4, drop=0.5, wd=1e-3, thr=0.3"
$CONDA_PYTHON scripts/train_risk_model.py $COMMON \
    --gpu 1 --run_name "sweep_heavy_reg" \
    --lr 1e-4 --dropout 0.5 --weight_decay 1e-3 --patience 10 --toxic_threshold 0.3 \
    --output "$CKPT_DIR/sweep_heavy_reg_w_risk_v2.pt" \
    --risk_output "$CKPT_DIR/sweep_heavy_reg_risk_model.pt" \
    2>&1 | tee "$LOG_DIR/sweep_heavy_reg.log" &

echo "🚀 [GPU 2] slow_learn: lr=5e-5, drop=0.4, wd=5e-4, thr=0.3"
$CONDA_PYTHON scripts/train_risk_model.py $COMMON \
    --gpu 2 --run_name "sweep_slow_learn" \
    --lr 5e-5 --dropout 0.4 --weight_decay 5e-4 --patience 10 --toxic_threshold 0.3 \
    --output "$CKPT_DIR/sweep_slow_learn_w_risk_v2.pt" \
    --risk_output "$CKPT_DIR/sweep_slow_learn_risk_model.pt" \
    2>&1 | tee "$LOG_DIR/sweep_slow_learn.log" &

echo "🚀 [GPU 3] big_patience: lr=1e-4, drop=0.4, wd=1e-4, thr=0.3, patience=20"
$CONDA_PYTHON scripts/train_risk_model.py $COMMON \
    --gpu 3 --run_name "sweep_big_patience" \
    --lr 1e-4 --dropout 0.4 --weight_decay 1e-4 --patience 20 --toxic_threshold 0.3 \
    --output "$CKPT_DIR/sweep_big_patience_w_risk_v2.pt" \
    --risk_output "$CKPT_DIR/sweep_big_patience_risk_model.pt" \
    2>&1 | tee "$LOG_DIR/sweep_big_patience.log" &

echo "🚀 [GPU 4] high_threshold: lr=1e-4, drop=0.3, wd=1e-4, thr=0.4"
$CONDA_PYTHON scripts/train_risk_model.py $COMMON \
    --gpu 4 --run_name "sweep_high_threshold" \
    --lr 1e-4 --dropout 0.3 --weight_decay 1e-4 --patience 10 --toxic_threshold 0.4 \
    --output "$CKPT_DIR/sweep_high_threshold_w_risk_v2.pt" \
    --risk_output "$CKPT_DIR/sweep_high_threshold_risk_model.pt" \
    2>&1 | tee "$LOG_DIR/sweep_high_threshold.log" &

echo ""
echo "⏳ Waiting for all 5 configs to finish..."
wait

echo ""
echo "══════════════════════════════════════════════"
echo "✅ All 5 configs finished! Extracting results..."
echo "══════════════════════════════════════════════"

for name in baseline heavy_reg slow_learn big_patience high_threshold; do
    echo ""
    echo "--- $name ---"
    grep -E "(TEST SET|test_mse|test_acc|test_recall|test_prec|checks)" "$LOG_DIR/sweep_${name}.log" 2>/dev/null || echo "(no results)"
done

echo ""
echo "🏆 Compare results above and pick the best model!"
echo "   Copy winner to: $CKPT_DIR/w_risk_v2.pt and $CKPT_DIR/risk_model.pt"

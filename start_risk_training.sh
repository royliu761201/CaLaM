#!/bin/bash
# ===========================================================================

# ===========================================================================

#

# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

CONDA_PYTHON="/root/miniconda3/envs/calam-unsloth/bin/python3"
CKPT_DIR="/jhdx0003008/data/calam/checkpoints"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

echo "=================================================="
echo "🧠 RiskModel Training Pipeline v1.0"
echo "   Checkpoint Dir: $CKPT_DIR"
echo "   GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo "=================================================="

# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "🔥 [Layer 1] Smoke Data Collection (N=50)..."
SMOKE_DATA_LOG="$LOG_DIR/risk_smoke_collect.log"
$CONDA_PYTHON scripts/collect_risk_data.py \
    --smoke \
    --output "$CKPT_DIR/risk_train_data_smoke.pt" \
    2>&1 | tee "$SMOKE_DATA_LOG"

SMOKE_COLLECT_EXIT=${PIPESTATUS[0]}
if [ $SMOKE_COLLECT_EXIT -ne 0 ]; then
    echo "❌ [FAIL-FAST] Smoke Data Collection Failed (exit=$SMOKE_COLLECT_EXIT)"
    echo "   Check logs: $SMOKE_DATA_LOG"
    exit 1
fi
echo "✅ [Layer 1] Smoke Data Collection通过！"

# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "🔥 [Layer 2] Smoke Training (3 epochs)..."
SMOKE_TRAIN_LOG="$LOG_DIR/risk_smoke_train.log"
$CONDA_PYTHON scripts/train_risk_model.py \
    --data "$CKPT_DIR/risk_train_data_smoke.pt" \
    --output "$CKPT_DIR/w_risk_v2_smoke.pt" \
    --risk_output "$CKPT_DIR/risk_model_smoke.pt" \
    --epochs 3 \
    --use_wandb \
    2>&1 | tee "$SMOKE_TRAIN_LOG"

SMOKE_TRAIN_EXIT=${PIPESTATUS[0]}
if [ $SMOKE_TRAIN_EXIT -ne 0 ]; then
    echo "❌ [FAIL-FAST] Smoke Training Failed (exit=$SMOKE_TRAIN_EXIT)"
    echo "   Check logs: $SMOKE_TRAIN_LOG"
    exit 1
fi
echo "✅ [Layer 2] Smoke Training通过！"

# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "⚡ [Layer 3] Full Data Collection (N=30000, shuffle, v3 多位置)..."
FULL_COLLECT_LOG="$LOG_DIR/risk_full_collect_$(date +%Y%m%d_%H%M%S).log"
PYTHONUNBUFFERED=1 $CONDA_PYTHON scripts/collect_risk_data.py \
    --num_samples 30000 \
    --batch_size 32 \
    --output "$CKPT_DIR/risk_train_data.pt" \
    2>&1 | tee "$FULL_COLLECT_LOG"

FULL_COLLECT_EXIT=${PIPESTATUS[0]}
if [ $FULL_COLLECT_EXIT -ne 0 ]; then
    echo "❌ [FAIL-FAST] Full Data Collection Failed (exit=$FULL_COLLECT_EXIT)"
    exit 1
fi
echo "✅ Data Collection Finished!"

echo ""
echo "⚡ [Layer 3] Formal Training (max 50 epochs, early stopping)..."
FULL_TRAIN_LOG="$LOG_DIR/risk_full_train_$(date +%Y%m%d_%H%M%S).log"
PYTHONUNBUFFERED=1 $CONDA_PYTHON scripts/train_risk_model.py \
    --data "$CKPT_DIR/risk_train_data.pt" \
    --output "$CKPT_DIR/w_risk_v2.pt" \
    --risk_output "$CKPT_DIR/risk_model.pt" \
    --epochs 50 \
    --use_wandb \
    2>&1 | tee "$FULL_TRAIN_LOG"

FULL_TRAIN_EXIT=${PIPESTATUS[0]}
if [ $FULL_TRAIN_EXIT -ne 0 ]; then
    echo "❌ [FAIL-FAST] Formal Training Failed (exit=$FULL_TRAIN_EXIT)"
    exit 1
fi

echo ""
echo "=================================================="
echo "🎉 RiskModel Training Complete!"
echo "   Feature ckpt: $CKPT_DIR/w_risk_v2.pt"
echo "   Risk ckpt   : $CKPT_DIR/risk_model.pt"
echo "   Collect log: $FULL_COLLECT_LOG"
echo "   Train log: $FULL_TRAIN_LOG"
echo "=================================================="

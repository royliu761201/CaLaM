#!/bin/bash
# CALAM 14B DPO Recalibration Pipeline Runtime (NeurIPS 2026)
echo "[*] Standardizing Environment Parameters..."
echo "[*] Engaging DPO Recalibration Sweep (beta=0.05)..."
python scripts/run_engine.py --config dpo_enterprise_sweep --beta_penalty 0.05 --target_layer all

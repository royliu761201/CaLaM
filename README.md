<div align="center">
  <h1>🏢 CaLaM: Mitigating Institutional Entropy and Semantic Drift via DPO</h1>
  <h3>NeurIPS 2026 Submission (Anonymous)</h3>
</div>

> [!IMPORTANT]
> **DOUBLE-BLIND COMPLIANCE**: This repository contains the structural code layout for the NeurIPS benchmarking matrices. Raw proprietary data and full weights have been decoupled.

<p align="center">
  A 14B parameter model optimized via Direct Preference Optimization (DPO) to suppress institutional entropy and strict computational "Semantic Drift."
</p>

## 🚀 Key Scientific Features
- **Institutional Entropy & Drift Metric (IEDM)**: A mathematically rigorous $JSD^2$ distance function.
- **Surgical Constraint Penalty**: Through a precisely bounded $\beta = 0.05$ recalibration sweep.

## 🛠️ Environment Setup & Quickstart

```bash
conda env create -f environment.yml
conda activate calam

python scripts/run_engine.py --config dpo_sweep --beta_penalty 0.05
python scripts/run_engine.py --config dpo_sweep --beta_penalty 0.05
```

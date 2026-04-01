import os
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class CaLaMConfig:
    """
    CaLaM Global Configuration
    """
    # Model defaults

    base_model: str = os.getenv("CALAM_MODEL_PATH", "Qwen/Qwen2.5-14B-Instruct")
    max_new_tokens: int = 128  # toxicity tasks; instruction tasks use 256 in config

    device: str = "cuda"

    # Steering
    steering_keywords: list = field(default_factory=lambda: [
        "hate", "stupid", "idiot", "kill", "ugly", 
        "shut", "fuck", "damn", "suck", "crap"
    ])

    risk_keywords: list = field(default_factory=lambda: [
        "stupid", "idiot", "kill", "hate", "ugly"
    ])

    # Tracking
    use_wandb: bool = True
    wandb_project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "calam"))

# Create a global config instance
CONFIG = CaLaMConfig()

# ==========================================
# Experiment Matrix Registry
# ==========================================
EXPERIMENT_MATRIX = {
    # --- Vanilla Baselines ---
    "vanilla_rtp": {
        "method": "vanilla",
        "dataset": "rtp",
        "limit": 99441,
        "batch_size": 224,
        "desc": "Vanilla Baseline (RTP)",
    },

    "vanilla_mmlu": {
        "method": "vanilla",
        "dataset": "mmlu",
        "data_path": os.getenv("CALAM_DATA_PATH", "./data/mmlu_test.csv"),
        "limit": 5000,   # Paper Sec 5.1: held-out MMLU subset 5,000 prompts
        "batch_size": 172,  # [L20 Utility Baseline Final]
        "desc": "Vanilla Baseline (MMLU)",
    },
    "vanilla_tqa": {
        "method": "vanilla",
        "dataset": "tqa",
        "data_path": os.getenv("CALAM_DATA_PATH", "./data/TruthfulQA.csv"),
        "limit": 817,
        "batch_size": 224,  # [L20 Utility Baseline Final]
        "desc": "Vanilla Baseline (TruthfulQA, full)",
    },

    "vanilla_xstest": {
        "method": "vanilla",
        "dataset": "xstest",
        "data_path": os.getenv("CALAM_DATA_PATH", "./data/xstest_v2_prompts.csv"),
        "limit": 250,
        "batch_size": 96,
        "evaluator": "xstest_refusal",
        "feature_mode": "safety",
        "desc": "Vanilla Baseline (XSTest)",
    },

    # --- CaLaM Dynamic Steering ---
    "calam_mmlu_v2_full": {
        "method": "calam",
        "dataset": "mmlu",
        "data_path": "/jhdx0003008/data/calam/mmlu_test.csv",
        "lambda_init": 5.0,
        "limit": 5000,
        "batch_size": 172,  # [L20 Optimized V5]
        "max_new_tokens": 256,
        "dynamic_bt": True,
        "desc": "CaLaM V2.1 Full Model (MMLU)",
    },
    "calam_tqa_v2_full": {
        "method": "calam",
        "dataset": "tqa",
        "data_path": "/jhdx0003008/data/calam/TruthfulQA.csv",
        "lambda_init": 5.0,
        "limit": 817,
        "batch_size": 224,  # [L20 Optimized V5]
        "max_new_tokens": 256,
        "dynamic_bt": True,
        "desc": "CaLaM V2.1 Full Model (TruthfulQA)",
    },

    # --- CaLaM 2x2 Ablation Matrix (f/b variants) ---
    "calam_rtp_v2_fixed_b": {
        "method": "calam",
        "dataset": "rtp",
        "lambda_init": 5.0,
        "limit": 99441,
        "batch_size": 128,  # [L20 Optimized V4]
        "desc": "CaLaM V2.0 Baseline (Dynamic f_t, Fixed b=0.1)",
    },
    "calam_rtp_full_v2_dynamic": {
        "method": "calam",
        "dataset": "rtp",
        "lambda_init": 5.0,
        "limit": 99441,
        "batch_size": 224,  # [L20 Optimized V5]
        "dynamic_bt": True,
        "desc": "CaLaM V2.1 Full Model (Dynamic f_t + Dynamic b_t)",
    },

    # --- DExperts Baseline (Table 1) ---
    "dexperts_rtp_full": {
        "method": "dexperts",
        "dataset": "rtp",
        "limit": 99441,
        "batch_size": 64,
        "desc": "DExperts Baseline on full RTP (Liu et al., 2021)",
    },

    # --- PPLM Baseline (Table 1) ---
    "pplm_rtp_full": {
        "method": "pplm",
        "dataset": "rtp",
        "limit": 99441,
        "batch_size": 84,
        "pplm_step_size": 0.5,    # [BUG-FIX] was 0.03 → delta≈0 for 14B hidden dim
        "pplm_num_iter": 10,      # [BUG-FIX] was 3 → insufficient iterations
        "desc": "PPLM Baseline on full RTP (Dathathri et al., 2020)",
    },

    # --- Self-Debiasing Baseline (Table 2) ---
    "self_debias_rtp_full": {
        "method": "self_debias",
        "dataset": "rtp",
        "limit": 99441,
        "batch_size": 96,  # [L20 Optimized V4]
        "desc": "Self-Debiasing Baseline full RTP (Schick et al., 2021)",
    },

    # --- XSTest ---
    "calam_xstest_v2_full": {
        "method": "calam",
        "dataset": "xstest",
        "data_path": "/jhdx0003008/data/calam/xstest_v2_prompts.csv",
        "lambda_init": 5.0,
        "limit": 250,
        "batch_size": 96,  # [L20 Optimized V4]
        "dynamic_bt": True,
        "evaluator": "xstest_refusal",
        "feature_mode": "safety",
        "desc": "CaLaM V2.1 Full Model on XSTest",
    },

    # --- SALAD-Bench ---
    "calam_salad_v2_full": {
        "method": "calam",
        "dataset": "salad_bench",
        "data_path": "/jhdx0003008/data/calam/SafetyBench/opensource_data/test_en.json",
        "lambda_init": 5.0,
        "limit": 1000,
        "batch_size": 96,  # [L20 Optimized V4]
        "dynamic_bt": True,
        "evaluator": "salad_safety",
        "feature_mode": "safety",
        "desc": "CaLaM V2.1 Full Model on SALAD-Bench",
    },

    # --- Ablation: No Geometry ---
    "calam_ablation_no_geom_v2_full": {
        "method": "calam",
        "dataset": "rtp",
        "lambda_init": 5.0,
        "limit": 99441,
        "batch_size": 128,  # [L20 Optimized V4]
        "dynamic_bt": True,
        "ablation_no_geometry": True,
        "desc": "Ablation: Remove manifold geometry (V2.1)",
    },

    # --- Ablation: No Lookahead ---
    "calam_ablation_no_lookahead_v2_full": {
        "method": "calam",
        "dataset": "rtp",
        "lambda_init": 5.0,
        "limit": 99441,
        "batch_size": 128,  # [L20 Optimized V4]
        "dynamic_bt": True,
        "ablation_no_lookahead": True,
        "desc": "Ablation: Remove lookahead (V2.1)",
    },
}

# ==========================================
# Claim -> Experiment Key Mapping
# ==========================================
CLAIM_TO_EXPERIMENT_MAP = {
    "Table 1: CaLaM outperforms baselines on RTP toxicity": [
        "vanilla_rtp", "dexperts_rtp_full", "pplm_rtp_full", "self_debias_rtp_full", "calam_rtp_full_v2_dynamic"
    ],
    "Table 2: CaLaM maintains utility (MMLU & TruthfulQA)": [
        "vanilla_mmlu", "calam_mmlu_v2_full",
        "vanilla_tqa", "calam_tqa_v2_full"
    ],
    "Table 2: Low over-refusal and high safety on XSTest & SALAD-Bench": [
        "vanilla_xstest", "calam_xstest_v2_full",
        "calam_salad_v2_full"
    ],
    "Table 3: Ablation on Geometry constraint and Lookahead": [
        "calam_rtp_full_v2_dynamic",
        "calam_ablation_no_geom_v2_full",
        "calam_ablation_no_lookahead_v2_full",
        "calam_rtp_v2_fixed_b"
    ],
}

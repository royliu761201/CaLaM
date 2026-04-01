import argparse
import fcntl
import json
import logging
import os
import random
import sys
import time

# GPU isolation: parse --gpu BEFORE importing torch
# This is necessary because device_map={"": 0} uses the first visible GPU
# and conda run can strip env vars set in shell scripts
if '--gpu' in sys.argv:
    gpu_idx = sys.argv[sys.argv.index('--gpu') + 1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

# CUDA allocator fragmentation was a contributing factor during long eager-attention runs.
alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in alloc_conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
        part for part in [alloc_conf, "expandable_segments:True"] if part
    )

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    # [FIX] Skip unsloth entirely if --no_unsloth is on the command line
    #       Unsloth crashes at import time with NotImplementedError if GPU is misconfigured
    if '--no_unsloth' not in sys.argv:
        os.environ["HF_HUB_OFFLINE"] = "1" 
        from unsloth import FastLanguageModel
        HAS_UNSLOTH = True
    else:
        HAS_UNSLOTH = False
except Exception as e:  # Catch ALL exceptions, not just ImportError
    HAS_UNSLOTH = False
    print(f"DEBUG: Unsloth Import Failed: {e}")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.calam import CaLaMController
from src.risk import RiskModel
from src.features import ToxicityFeatureExtractor, SemanticFeatureExtractor, DynamicContextFeatureExtractor
from src.data import RealToxicityPromptsLoader, SafetyBenchLoader, TruthfulQALoader, JailbreakBenchLoader
from src.config import EXPERIMENT_MATRIX, CONFIG
from src.sample_selection import (
    extract_prompt_text,
    prompt_length,
    select_longest_samples,
    sort_samples_by_prompt_length,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SMOKE_WORST_CASE_SAMPLE_COUNT = 50
DEFAULT_PROMPT_MAX_LENGTH = 512
DEFAULT_ACCURACY_PREFILL_MAX_BATCH_AT_MAX_LENGTH = 48
DEFAULT_ACCURACY_PREFILL_BUDGET = (
    DEFAULT_ACCURACY_PREFILL_MAX_BATCH_AT_MAX_LENGTH * (DEFAULT_PROMPT_MAX_LENGTH ** 2)
)

def prepare_evaluation_samples(samples, *, limit, smoke, start_index):
    selected_samples = list(samples)

    if limit is not None:
        logger.info(f"🛑 Limit Override: Truncating dataset to first {limit} samples.")
        selected_samples = selected_samples[:limit]
    elif smoke:
        selected_samples = select_longest_samples(selected_samples, SMOKE_WORST_CASE_SAMPLE_COUNT)
        if selected_samples:
            logger.info(
                "🔥 Smoke Test [Worst-Case Profiling]: Selected %s longest samples "
                "(max_chars=%s, min_chars=%s).",
                len(selected_samples),
                prompt_length(selected_samples[0]),
                prompt_length(selected_samples[-1]),
            )
        else:
            logger.info("🔥 Smoke Test [Worst-Case Profiling]: Dataset is empty after selection.")

    if start_index >= len(selected_samples):
        logger.warning(
            "Requested start_index (%s) already reached dataset boundary (%s).",
            start_index,
            len(selected_samples),
        )

    remaining_samples = selected_samples[start_index:]
    if remaining_samples and not smoke:
        logger.info(
            "📦 [Smart Batching] Sorting remaining dataset by prompt length to minimize padding overhead."
        )
        remaining_samples = sort_samples_by_prompt_length(remaining_samples)

    return selected_samples, remaining_samples

def estimate_truncated_prompt_lengths(tokenizer, prompts, *, max_length):
    encoded = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]

def resolve_adaptive_batch_size(samples, *, tokenizer, base_batch_size, max_length, attention_budget):
    prompts = [extract_prompt_text(sample, strict=True) for sample in samples]
    lengths = estimate_truncated_prompt_lengths(tokenizer, prompts, max_length=max_length)
    max_tokens = max(lengths) if lengths else 1
    avg_tokens = (sum(lengths) / len(lengths)) if lengths else 0.0
    safe_batch_size = max(1, int(attention_budget // (max_tokens * max_tokens)))
    effective_batch_size = min(len(samples), base_batch_size, safe_batch_size)
    return effective_batch_size, max_tokens, avg_tokens

def main():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    parser = argparse.ArgumentParser(description="CaLaM Experiment Runner")
    parser.add_argument("--task", type=str, help="Task ID from EXPERIMENT_MATRIX")
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID (sets CUDA_VISIBLE_DEVICES before torch init)")
    parser.add_argument("--lambda_init", type=float, default=None, help="Initial lambda value")
    parser.add_argument("--mock", action="store_true", help="Use mock model/tokenizer for testing")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--max_memory_per_gpu", type=str, default=None, help="Max GPU memory (e.g. 40GiB)")
    # --batch_size REMOVED: config.py EXPERIMENT_MATRIX is the sole SSoT for BS (Boss Directive 2026-03-26)
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Enable W&B Logging")
    parser.add_argument("--no_unsloth", action="store_true", help="Force disable Unsloth")

    # Extra flags from run_remote_gpu
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no control)")
    parser.add_argument("--ablation_no_geometry", action="store_true", help="Ablation: Disable curvature awareness")
    parser.add_argument("--ablation_no_lookahead", action="store_true", help="Ablation: Disable lookahead control")
    parser.add_argument("--constraint_threshold", type=float, default=0.1, help="Safety constraint threshold")
    parser.add_argument("--dynamic_bt", action="store_true", default=False, help="Enable dynamic b_t = g(risk) constraint")
    parser.add_argument("--baseline_correction", action="store_true", help="Baseline: Auto-Correction (Rainier-style)")
    parser.add_argument("--use_pid", action="store_true", default=True, help="Use PID controller for Dual Ascent (Default: True for robustness)")
    parser.add_argument("--show_reasoning", action="store_true", help="Show reasoning traces (skip_special_tokens=False)")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for resuming")
    parser.add_argument("--force_restart", action="store_true", help="Force delete old logs and start from zero")
    parser.add_argument("--dexperts", action="store_true", help="Run DExperts Baseline instead of CaLaM")
    parser.add_argument("--dexperts_alpha", type=float, default=2.0, help="DExperts alpha strength")
    parser.add_argument("--pplm", action="store_true", help="Run PPLM Baseline")
    parser.add_argument("--self_debias", action="store_true", help="Run Self-Debiasing Baseline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (Overrides --smoke if provided)")
    # --config is an alias for --task (used by omni-scheduler enqueue commands)
    parser.add_argument("--config", type=str, default=None, help="Alias for --task")
    args = parser.parse_args()
    if not args.task and not args.config:
        raise ValueError("--task or --config is required.")

    # Alias resolution: --config wins if --task not set
    if args.config and not args.task:
        args.task = args.config

    # 0. Reproducibility + Global CUDA Optimizations
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True  # [PERF] Auto-tune conv kernels
        torch.backends.cuda.matmul.allow_tf32 = True  # [PERF] TF32 for matmul
        torch.backends.cudnn.allow_tf32 = True  # [PERF] TF32 for cudnn

    # Load Config from Task

    # Load Config from Task
    task_conf = {}
    if args.task:
        if args.task not in EXPERIMENT_MATRIX:
            raise ValueError(f"[FAIL-FAST 🚨] The task alias '{args.task}' is NOT found in EXPERIMENT_MATRIX! Please check src/config.py to avoid silent default fallback.")

        logger.info(f"Loading config for task: {args.task}")
        task_conf = EXPERIMENT_MATRIX[args.task]
        # Override args if not provided
        if args.model is None: args.model = task_conf.get("model", "/jhdx0003008/models/Qwen2.5-14B-Instruct")
        if args.data_path is None: args.data_path = task_conf.get("data_path")
        if args.lambda_init is None: args.lambda_init = task_conf.get("lambda_init", 0.5)
        if args.device is None: args.device = CONFIG.device
        if task_conf.get("mock", False): args.mock = True
        if task_conf.get("method") == "pplm": args.pplm = True
        if task_conf.get("method") == "self_debias": args.self_debias = True
        if task_conf.get("method") == "dexperts": args.dexperts = True  # [BUG-FIX] was missing!
        if task_conf.get("method") == "vanilla": args.baseline = True  # [PERF] vanilla = baseline, enables MMLU shortcut
        # [BUG-FIX] Ablation flags were defined in config but NEVER propagated to args!
        if task_conf.get("ablation_no_geometry"): args.ablation_no_geometry = True
        if task_conf.get("ablation_no_lookahead"): args.ablation_no_lookahead = True
        # Config-level overrides for experiment parameters
        if task_conf.get("constraint_threshold") is not None: args.constraint_threshold = task_conf["constraint_threshold"]
        if task_conf.get("dynamic_bt"): args.dynamic_bt = True

        # [AB-047] Pre-flight assertions: verify config flags are actually propagated
        for key in ["ablation_no_geometry", "ablation_no_lookahead"]:
            if task_conf.get(key) and not getattr(args, key, False):
                raise RuntimeError(f"[AB-047] FATAL: Config key '{key}' is True but args.{key} is False! Config-Args breakage detected.")

    # Defaults
    if args.model is None or args.model == "gpt2" or "Qwen/" in args.model:
        args.model = "/jhdx0003008/models/Qwen2.5-14B-Instruct" 

    if args.device is None: args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.lambda_init is None: args.lambda_init = 0.5

    # Mock Classes
    class MockEncoding(dict):
        def to(self, device):
            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    self[k] = v.to(device)
            return self

    class MockTokenizer:
        pad_token = "[PAD]"
        def __call__(self, text, return_tensors="pt", **kwargs):
            # Return dummy input_ids (1, 5)
            data = {"input_ids": torch.randint(0, 100, (1, 5)), "attention_mask": torch.ones(1, 5)}
            return MockEncoding(data)
        def decode(self, ids, skip_special_tokens=True):
            return "mock decoded text"
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=768):
            super().__init__()
            self.vocab_size = vocab_size
            self.config = type('Config', (), {'hidden_size': hidden_size})()
            self.linear = torch.nn.Linear(hidden_size, vocab_size)
        def forward(self, input_ids, output_hidden_states=False, **kwargs):
            B, L = input_ids.shape

            # Artificial Heavy GPU Load for benchmarking
            if input_ids.device.type == "cuda":
                for _ in range(50):
                    matrix = torch.randn(4096, 4096, device=input_ids.device)
                    _ = torch.matmul(matrix, matrix)

            logits = torch.randn(B, L, self.vocab_size, device=input_ids.device)
            out = type('Output', (), {'logits': logits, 'past_key_values': None})()
            if output_hidden_states:
                out.hidden_states = (torch.randn(B, L, self.config.hidden_size, device=input_ids.device),)
            return out
        def get_input_embeddings(self):
            # Return dummy embedding layer
            return torch.nn.Embedding(self.vocab_size, self.config.hidden_size)
        def to(self, device):
            return self
        def eval(self):
            pass

    # Initialize W&B (Global)
    if hasattr(args, 'use_wandb') and args.use_wandb:
        logger.info(f"📊 Initializing W&B for task: {args.task}")
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "calam"),
            name=args.task,
            config=vars(args),
            group="v2_fixed_20260318",
            tags=["v2_fixed", "offline_data", "fixed_alpha"]
        )
        wandb.config.update({
            "jit_enabled": (not args.baseline and not args.dexperts and not args.pplm and not args.self_debias),
            "warmstart_enabled": (not args.baseline and not args.dexperts and not args.pplm and not args.self_debias),
        })

    # Load Model & Tokenizer

    if args.mock:
        raise RuntimeError(
            "[FAIL-FAST] --mock mode is PROHIBITED. "
            "Use --smoke (N=50 real data) for pipeline validation."
        )
    elif HAS_UNSLOTH and args.load_in_4bit and not args.no_unsloth:
        logger.info(f"🚀 Loading via Unsloth (Optimized 4-bit): {args.model}")
        max_seq_length = 4096
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            device_map={"": 0},
            local_files_only=True
        )
        model.config.output_hidden_states = True
        # [Phase C] Unsloth inference optimization — without this, Unsloth speedup is ZERO
        FastLanguageModel.for_inference(model)
        logger.info("⚡ [Phase C] Unsloth for_inference() activated — Triton kernels + fused ops enabled")
    else:
        logger.info(f"⚠️ Unsloth NOT used. Reasons: HAS_UNSLOTH={HAS_UNSLOTH}, 4bit={args.load_in_4bit}, NoUnsloth={args.no_unsloth}")
        logger.info(f"Loading model (Transformers): {args.model}")

        # Detect flash_attn availability
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
            logger.info("✅ flash_attn available — using flash_attention_2")
        except ImportError:
            _attn_impl = "eager"
            logger.info("⚠️ flash_attn not installed — falling back to eager attention")

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

        # Quantization Logic
        bnb_config = None
        if args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif args.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Max Memory Logic
        max_memory_map = None
        if args.max_memory_per_gpu:
            max_memory_map = {0: args.max_memory_per_gpu, "cpu": "100GiB"}
            logger.info(f"Enforcing max_memory: {max_memory_map}")

        if bnb_config:
            logger.info(f"Loading model with quantization: 4bit={args.load_in_4bit}, 8bit={args.load_in_8bit}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                quantization_config=bnb_config, 
                device_map="auto",
                max_memory=max_memory_map, 
                attn_implementation=_attn_impl,
                local_files_only=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                device_map="auto", 
                attn_implementation=_attn_impl,
                torch_dtype=torch.bfloat16,  # [PERF] BF16 inference
                local_files_only=True
            )

        # Unsloth handles eval() internally in for_inference, but good to be explicit for transformers
        model.eval()
        # [Phase A] torch.compile — DISABLED
        # CUDAGraphs (reduce-overhead mode) crashes in multi-process parallel experiments
        # RuntimeError: accessing tensor output of CUDAGraphs overwritten by subsequent run
        
        logger.info("⚠️ [Phase A] torch.compile DISABLED (CUDAGraphs multi-process conflict)")

    # Initialize CaLaM or Baselines
    calam = None
    dexperts_ctrl = None
    self_debias_ctrl = None
    pplm_ctrl = None

    if args.dexperts:
        from src.dexperts import DExpertsController
        dexperts_ctrl = DExpertsController(
            device=args.device,
            alpha=args.dexperts_alpha,
            load_in_4bit=True  # Force 4bit: base(bf16 ~28G) + expert(bf16 ~28G) = 56G > 48G per card
        )
    elif args.self_debias:
        from src.self_debias import SelfDebiasingController
        self_debias_ctrl = SelfDebiasingController(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            alpha=1.5  # Typical alpha for self-debiasing
        )
    elif args.pplm:
        # pplm_ctrl needs risk_model & embeddings; we'll instantiate it after risk_model init.
        pass
    elif not args.baseline:
        # num_features=1 for simple toxicity constraint
        calam = CaLaMController(
            num_features=1, 
            lambda_init=args.lambda_init,
            device=args.device
        )

    # [PERF] Baseline runs do not need auxiliary control modules or a duplicated embedding matrix on GPU.
    embedding_matrix = None
    risk_model = None
    feature_extractor = None

    if not args.baseline:
        embedding_matrix = model.get_input_embeddings().weight

        if "_v2" in args.task:
            feature_extractor = DynamicContextFeatureExtractor(
                hidden_size=model.config.hidden_size,
                embedding_matrix=embedding_matrix,
                device=args.device
            )
            risk_ckpt = "/jhdx0003008/data/calam/checkpoints/w_risk_v2.pt"
            if not os.path.exists(risk_ckpt):
                raise FileNotFoundError(f"Required V2 feature checkpoint missing: {risk_ckpt}")
            feature_extractor.load_state_dict(torch.load(risk_ckpt, map_location=args.device), strict=True)
            feature_extractor.eval()
            logger.info(f"✅ Loaded V2 Feature Extractor (W_risk) from {risk_ckpt}")

        if task_conf.get("dynamic_bt", False) or args.pplm or "_v2" in args.task:
            from src.risk import RiskModel
            risk_model = RiskModel(hidden_size=model.config.hidden_size).to(args.device)
            risk_ckpt = "/jhdx0003008/data/calam/checkpoints/risk_model.pt"
            if not os.path.exists(risk_ckpt):
                raise FileNotFoundError(f"Required risk checkpoint missing: {risk_ckpt}")
            risk_model.load_state_dict(torch.load(risk_ckpt, map_location=args.device), strict=True)
            logger.info(f"✅ Loaded Dynamic Risk Model (b_t selector) from {risk_ckpt}")
            risk_model.eval()

        if args.pplm:
            from src.pplm import PPLMController
            pplm_step_size = task_conf.get("pplm_step_size", 0.5)
            pplm_num_iter = task_conf.get("pplm_num_iter", 10)
            pplm_ctrl = PPLMController(
                risk_model=risk_model, # [FIXED] Sourcing from initialized risk_model
                embedding_matrix=embedding_matrix,
                device=args.device,
                step_size=pplm_step_size,
                num_iter=pplm_num_iter
            )
            logger.info(f"🔧 PPLM initialized: step_size={pplm_step_size}, num_iter={pplm_num_iter}")

        if "_v2" in args.task:
            pass
        else:
            # V1.x Static Semantic Feature Extractor
            from src.features import SemanticFeatureExtractor
            feature_extractor = SemanticFeatureExtractor(
                embedding_matrix=embedding_matrix,
                tokenizer=tokenizer,
                device=args.device
            )
            feature_mode = task_conf.get("feature_mode", "toxicity")
            feature_extractor.set_mode(feature_mode)
            logger.info(f"✅ SemanticFeatureExtractor (V1 Static) instantiated.")
    else:
        logger.info("⚡ Baseline mode: skipping feature extractor / risk model initialization.")

    # Load Data
    logger.info(f"Loading data...")
    if "safetybench" in (args.data_path or "").lower():
        dataset = SafetyBenchLoader(args.data_path)
    elif "truthfulqa" in (args.data_path or "").lower() or (args.task and "tqa" in args.task.lower()):
        dataset = TruthfulQALoader(args.data_path or "/jhdx0003008/data/calam/TruthfulQA.csv")
    elif "jailbreak" in (args.data_path or "").lower():
        dataset = JailbreakBenchLoader(args.data_path)
    elif "xstest" in (args.data_path or "").lower() or (args.task and "xstest" in args.task.lower()):
        from src.data import XSTestLoader
        dataset = XSTestLoader(args.data_path)
    elif "mmlu" in (args.data_path or "").lower() or "mmlu" in (args.task or "").lower():
        from src.data import MMLULoader
        dataset = MMLULoader(args.data_path)
    else:
        dataset = RealToxicityPromptsLoader(args.data_path or "/jhdx0003008/data/calam/realtoxicityprompts-data.jsonl")

    if isinstance(dataset, RealToxicityPromptsLoader):
        dataset.load(challenging_only=task_conf.get("challenging_only", False))
    else:
        dataset.load()

    # Apply experiment-level limit BEFORE start_index to respect EXPERIMENT_MATRIX config
    exp_limit = task_conf.get("limit", None)
    if exp_limit:
        dataset.samples = dataset.samples[:exp_limit]
        logger.info(f"[EXPERIMENT_MATRIX] Limiting to {exp_limit} samples as per config.")

    journal_prefix = "smoke_" if args.smoke else ""
    journal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", f"{journal_prefix}{args.task}_journal.jsonl")
    lock_path = journal_path + ".lock"
    all_toxicity_scores = []
    all_accuracy_scores = []
    is_accuracy_task = (args.task and any(k in args.task.lower() for k in ["mmlu", "tqa"]))

    override_start = False

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            if os.path.exists(journal_path) and getattr(args, "force_restart", False):
                open(journal_path, "w").close() # Safe truncate strictly under global lock
                args.start_index = 0
                logger.warning(f"⚠️ --force_restart passed. Truncated old journal: {journal_path}")
            elif os.path.exists(journal_path) and args.start_index == 0:
                logger.info(f"🔄 Auto-Resuming from JSONL Journal: {journal_path}")
                max_idx = -1
                with open(journal_path, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            idx = data.get("global_idx", -1)
                            if idx > max_idx: max_idx = idx
                            score = data.get("score")
                            if score is not None:
                                if is_accuracy_task:
                                    all_accuracy_scores.append(score)
                                else:
                                    all_toxicity_scores.append(score)
                        except Exception:
                            pass
                args.start_index = max_idx + 1 if max_idx >= 0 else 0
                rec_len = len(all_accuracy_scores) if is_accuracy_task else len(all_toxicity_scores)
                logger.info(f"✅ Recovered {rec_len} historical scores. Auto-setting start_index to {args.start_index}.")
            elif args.start_index > 0:
                logger.info(f"⚠️ Using explicit CLI start_index: {args.start_index}")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    dataset.samples, all_samples = prepare_evaluation_samples(
        dataset.samples,
        limit=args.limit,
        smoke=args.smoke,
        start_index=args.start_index,
    )

    logger.info(f"Starting CaLaM evaluation loop... Processing {len(all_samples)} samples.")

    # [SSoT] Batch Size — config.py EXPERIMENT_MATRIX is the ONLY source (Boss Directive 2026-03-26)
    batch_size = task_conf["batch_size"]
    logger.info(f"📐 [SSoT] batch_size={batch_size} (from config.py EXPERIMENT_MATRIX)")
    prompt_max_length = task_conf.get("prompt_max_length", DEFAULT_PROMPT_MAX_LENGTH)
    adaptive_attention_budget = None
    if is_accuracy_task:
        adaptive_attention_budget = task_conf.get(
            "prefill_attention_budget",
            DEFAULT_ACCURACY_PREFILL_BUDGET,
        )
        logger.info(
            "🧠 [Adaptive BS] Accuracy prefill budget=%s tokens^2, prompt_max_length=%s",
            adaptive_attention_budget,
            prompt_max_length,
        )

    def batch_collate(samples):
        prompts = []
        labels = []
        for s in samples:
            p = extract_prompt_text(s, strict=True)
            prompts.append(p)

            if isinstance(s, dict) and "answer" in s:
                labels.append(s["answer"])
            else:
                labels.append(None)

        # Tokenize with Left Padding for Generation
        tokenizer.padding_side = "left" 
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        )
        return inputs, prompts, labels

    # Generator
    def chunked_data(data, bs):
        idx = 0
        while idx < len(data):
            candidate = data[idx:idx+bs]
            effective_bs = min(len(candidate), bs)

            if adaptive_attention_budget is not None and candidate:
                effective_bs, max_tokens, avg_tokens = resolve_adaptive_batch_size(
                    candidate,
                    tokenizer=tokenizer,
                    base_batch_size=bs,
                    max_length=prompt_max_length,
                    attention_budget=adaptive_attention_budget,
                )
                if effective_bs < len(candidate):
                    logger.info(
                        "🪓 [Adaptive BS] Reducing batch from %s to %s (max_tokens=%s, avg_tokens=%.1f, budget=%s).",
                        len(candidate),
                        effective_bs,
                        max_tokens,
                        avg_tokens,
                        adaptive_attention_budget,
                    )

            yield candidate[:effective_bs]
            idx += effective_bs

    max_new_tokens = task_conf.get("max_new_tokens", 128)

    temperature = task_conf.get("temperature", 0.7)
    top_p = task_conf.get("top_p", 0.9)
    logger.info(f"[PAPER ALIGN] max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
    start_time = time.time()
    total_processed = args.start_index

    evaluator = task_conf.get("evaluator", "toxicity")
    scorer = None
    if is_accuracy_task:
        logger.info("⚡ Accuracy task detected — skipping ToxicityScorer initialization.")
    elif evaluator == "toxicity" and not args.mock:
        try:
            from src.scorer import ToxicityScorer
            scorer = ToxicityScorer()
            logger.info("[AB-037] ToxicityScorer loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"[AB-038] FATAL: ToxicityScorer failed to load: {e}. "
                "Fix model path /jhdx0003008/models/toxic-bert before running."
            ) from e
    elif evaluator in {"xstest_refusal", "salad_safety"}:
        from src.scorer import DummyScorer
        scorer = DummyScorer()
        logger.info(f"[AB-037] {evaluator} uses DummyScorer to flush generations to journal for post-hoc LLM evaluation.")
    else:
        logger.warning("[MOCK] Mock mode: ToxicityScorer disabled. DO NOT use mock results in paper.")

    class LastHiddenHook:
        def __init__(self):
            self.last_hidden = None
        def __call__(self, module, inputs, output):
            if isinstance(output, tuple):
                self.last_hidden = output[0]
            else:
                self.last_hidden = output

    hook = LastHiddenHook()
    last_layer = None

    if not args.baseline:
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                last_layer = model.model.layers[-1]
            elif hasattr(model, "layers"):
                last_layer = model.layers[-1]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                last_layer = model.transformer.h[-1]
        except Exception as e:
            logger.error(f"❌ Failed to find last layer: {e}")

    if last_layer:
        last_layer.register_forward_hook(hook)
        logger.info("✅ Forward Hook registered for Hidden States.")
    elif args.baseline:
        logger.info("⚡ Baseline mode: hidden-state hook disabled.")
    else:
        logger.warning("⚠️ Hook not registered — fallback to output_hidden_states=True.")

    if is_accuracy_task:
        logger.info("⚡ [ACCURACY SHORTCUT] MMLU/TQA detected — using single-token logit extraction (A/B/C/D), skipping 128-step generation.")

    for batch_samples in chunked_data(all_samples, batch_size):
        inputs, prompts, labels = batch_collate(batch_samples)
        input_ids = inputs["input_ids"].to(args.device)
        attn_mask = inputs["attention_mask"].to(args.device)

        B = input_ids.size(0)

        # KV Cache Setup
        past_key_values = None
        curr_ids = input_ids

        sample_start_time = time.time()  # [PERF] Per-batch latency tracking

        # [C3 FIX] MMLU is always single-token accuracy — works for ALL methods
        # However, we MUST apply the control interventions to the logits before taking argmax!
        if is_accuracy_task:
            with torch.no_grad():
                # [PERF] logits_to_keep=1 avoids allocating (B, L, V) tensor.
                need_hidden_state = (not args.baseline)
                use_hook = need_hidden_state and (last_layer is not None)
                try:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        return_dict=True,
                        output_hidden_states=(need_hidden_state and not use_hook),
                        logits_to_keep=1,
                    )
                    next_token_logits = outputs.logits[:, -1, :]  # (B, vocab_size)
                except TypeError:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        return_dict=True,
                        output_hidden_states=(need_hidden_state and not use_hook),
                    )
                    next_token_logits = outputs.logits[:, -1, :]  # (B, vocab_size)

                # Apply Control Interventions before ABCD extraction
                if not args.baseline:
                    if use_hook and hook.last_hidden is not None:
                        raw = hook.last_hidden
                        hook.last_hidden = None
                        last_hidden = raw[:, -1, :].clone().to(args.device)
                        del raw
                    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        last_hidden = outputs.hidden_states[-1][:, -1, :].to(args.device)
                    else:
                        raise RuntimeError(
                            "Accuracy fast path failed to capture last hidden state; "
                            "refusing zero-vector fallback."
                        )

                    del outputs  # [PERF] Early buffer claim

                    # [V2.0/V1] Dynamic routing based on extractor type
                    if isinstance(feature_extractor, DynamicContextFeatureExtractor):
                        features = feature_extractor(last_hidden).to(next_token_logits.dtype)
                    else:
                        features = feature_extractor(input_ids, next_token_logits).to(next_token_logits.dtype)
                    b_max = task_conf.get("constraint_threshold", 0.1)
                    if risk_model is not None and task_conf.get("dynamic_bt", False):
                        with torch.no_grad():
                            risk = risk_model(last_hidden)  # (B,)
                            b_t = ((1.0 - risk) * b_max).clamp(min=1e-3, max=b_max)
                            constraints = b_t.unsqueeze(-1) # (B, 1)
                    else:
                        constraints = torch.tensor([[b_max]] * B, device=args.device)

                    if args.dexperts:
                        final_logits, _, _ = dexperts_ctrl.control(
                            next_token_logits, curr_ids, attention_mask=attn_mask,
                            expert_past_key_values=None, antiexpert_past_key_values=None)
                    elif args.self_debias:
                        final_logits, _ = self_debias_ctrl.control(
                            next_token_logits, curr_ids, toxic_past_key_values=None, attention_mask=attn_mask)
                    elif args.pplm:
                        final_logits = pplm_ctrl.control(next_token_logits, last_hidden)
                    else:
                        prev_lam = torch.full((B, calam.m if calam else 1), fill_value=task_conf.get("lambda_init", 0.0), device=args.device)
                        final_logits, _ = calam.control(
                            next_token_logits, features, constraints,
                            use_geometry=(not args.ablation_no_geometry),
                            use_lookahead=(not args.ablation_no_lookahead),
                            prev_lam=prev_lam
                        )
                else:
                    final_logits = next_token_logits
                    del outputs

                # ABCD token IDs for Qwen/Llama family
                abcd_tokens = tokenizer.encode("A B C D", add_special_tokens=False)
                abcd_ids = [t for t in abcd_tokens if tokenizer.decode([t]).strip() in "ABCD"]
                if len(abcd_ids) == 4:
                    abcd_logits = final_logits[:, abcd_ids]  # (B, 4)
                    pred_idx = abcd_logits.argmax(dim=-1)  # (B,)
                    pred_labels = ["A", "B", "C", "D"]
                    batch_texts = [pred_labels[idx.item()] for idx in pred_idx]
                else:
                    # Fallback: greedy decode 1 token
                    next_token = final_logits.argmax(dim=-1, keepdim=True)
                    batch_texts = [tokenizer.decode(t, skip_special_tokens=True).strip() for t in next_token]

                for b in range(B):
                    global_idx = total_processed + b
                    gt = labels[b] if labels[b] is not None else ""
                    print(f"Example {global_idx}: Pred={batch_texts[b]} GT={gt}")

                # Score
                if is_accuracy_task:
                    # Accuracy scoring
                    for idx, pred in enumerate(batch_texts):
                        gt = labels[idx] if labels[idx] is not None else ""
                        score = 1.0 if pred.strip() == gt.strip() else 0.0
                        all_accuracy_scores.append(score)
                        with open(journal_path, "a") as jf:
                            fcntl.flock(jf, fcntl.LOCK_EX)
                            jf.write(json.dumps({
                                "global_idx": total_processed + idx,
                                "score": score,
                                "prompt": prompts[idx][:200],
                                "generated": pred,
                                "gt": gt,
                                "latency_s": round(time.time() - sample_start_time, 3)
                            }) + "\n")
                elif scorer is not None:
                    # Original toxicity scoring
                    batch_scores = scorer.score(batch_texts)
                    for idx, score in enumerate(batch_scores):
                        all_toxicity_scores.append(score)
                        with open(journal_path, "a") as jf:
                            fcntl.flock(jf, fcntl.LOCK_EX)
                            jf.write(json.dumps({
                                "global_idx": total_processed + idx,
                                "score": score,
                                "prompt": prompts[idx][:200],
                                "generated": batch_texts[idx],
                                "latency_s": round(time.time() - sample_start_time, 3)
                            }) + "\n")

                total_processed += B
                elapsed = time.time() - start_time
                speed = (total_processed - args.start_index) / elapsed
                total_samples = len(dataset.samples)
                remaining = total_samples - total_processed
                eta_h = (remaining / speed) / 3600 if speed > 0 else 0
                logger.info(f"🚀 Processed {total_processed}/{total_samples} (Speed: {speed:.2f} samples/s | ETA: {eta_h:.1f}h)")
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({"speed_samples_per_sec": speed, "eta_hours": eta_h})
                continue  # Skip the full generation loop below

        # Pre-fill Phase
        with torch.no_grad():
            # MEMORY OPTIMIZATION: Do NOT request output_hidden_states=True if hook is active
            need_hidden_state = (not args.baseline)
            use_hook = need_hidden_state and (last_layer is not None)

            try:
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attn_mask, 
                    use_cache=True, 
                    output_hidden_states=(need_hidden_state and not use_hook), # Only request if hook failed
                    return_dict=True,
                    logits_to_keep=1
                )
            except TypeError:
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attn_mask, 
                    use_cache=True, 
                    output_hidden_states=(need_hidden_state and not use_hook),
                    return_dict=True
                )

            if hasattr(outputs, 'past_key_values'):
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :].clone().to(args.device) 

                if need_hidden_state:
                    if use_hook and hook.last_hidden is not None:
                        # Retrieve from Hook
                        raw = hook.last_hidden
                        hook.last_hidden = None # Clear immediately
                        last_hidden = raw[:, -1, :].clone().to(args.device)
                        del raw
                    elif outputs.hidden_states is not None:
                        last_hidden = outputs.hidden_states[-1][:, -1, :].clone().to(args.device)
                    else:
                        logger.error("❌ [P0 CRITICAL] Hidden state hook FAILED to capture output!")
                        raise RuntimeError("Dynamic control FAILED: Hidden state hook returned None. Check model version or disable control.")
                del outputs  # [PERF] Drop prefill outputs
            else:
                # Tuple Fallback
                 # ... (Existing logic for tuple fallback if needed, or assume dict for simplicity in optimization)
                 pass

            # Initial Risk/Features (based on prompt end)
            # [PERF] Skip heavy feature extraction for vanilla baseline
            if not args.baseline:
                logger.debug(f"Input shape: {last_hidden.shape}")
                # [V2.0/V1] Dynamic routing based on extractor type
                if isinstance(feature_extractor, DynamicContextFeatureExtractor):
                    features = feature_extractor(last_hidden).to(next_token_logits.dtype)
                else:
                    features = feature_extractor(input_ids, next_token_logits).to(next_token_logits.dtype)

        # Storage for generated sequence
        generated_tokens = []
        prev_lam = torch.full((B, calam.m if calam else 1), fill_value=task_conf.get("lambda_init", 0.0), device=args.device)
        lambda_trajectory = []  # [Case Study] λ(t) per generation step

        expert_past = None
        antiexpert_past = None
        toxic_past = None

        # [PERF] Precompute seq_lens to avoid attn_mask.sum() looping
        seq_lens = attn_mask.sum(dim=-1).unsqueeze(-1)

        # Generation Loop
        for step in range(max_new_tokens):
            with torch.no_grad():
                # [V2.1] Dynamic constraint: b_t = risk * b_max
                b_max = task_conf.get("constraint_threshold", 0.1)
                if risk_model is not None and last_hidden is not None and task_conf.get("dynamic_bt", False):
                    risk = risk_model(last_hidden)  # (B,)
                    b_t = ((1.0 - risk) * b_max).clamp(min=1e-3, max=b_max)
                    constraints = b_t.unsqueeze(-1) # (B, 1)
                else:
                    constraints = torch.full((B, 1), fill_value=b_max, device=args.device, dtype=next_token_logits.dtype)

                # Control
                if args.baseline:
                    final_logits = next_token_logits
                elif args.dexperts:
                    final_logits, expert_past, antiexpert_past = dexperts_ctrl.control(
                        next_token_logits, curr_ids,
                        attention_mask=attn_mask,
                        expert_past_key_values=expert_past,
                        antiexpert_past_key_values=antiexpert_past,
                    )
                elif args.self_debias:
                    final_logits, toxic_past = self_debias_ctrl.control(
                        next_token_logits, curr_ids, toxic_past_key_values=toxic_past, attention_mask=attn_mask
                    )
                elif args.pplm:
                    final_logits = pplm_ctrl.control(next_token_logits, last_hidden)
                else:
                    final_logits, prev_lam = calam.control(
                        next_token_logits,
                        features,
                        constraints,
                        use_geometry=(not args.ablation_no_geometry),
                        use_lookahead=(not args.ablation_no_lookahead),
                        prev_lam=prev_lam
                    )
                    # [Case Study] Record λ(t)
                    lam_mean = prev_lam.mean().item()
                    lambda_trajectory.append(lam_mean)

                # Defensive squeeze: some control methods may return 3D logits
                if final_logits.dim() > 2:
                    final_logits = final_logits.squeeze(1)  # (B, 1, V) → (B, V)
                scaled_logits = final_logits / temperature
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                # Top-p (nucleus) sampling
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens with cumulative prob above top_p (shift right to keep first above)
                sorted_remove = (cumulative_probs - sorted_probs) > top_p
                sorted_probs[sorted_remove] = 0.0
                prob_sums = sorted_probs.sum(dim=-1, keepdim=True)
                sorted_probs = sorted_probs / prob_sums.clamp(min=1e-8)
                next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_idx.gather(-1, next_token_sorted)  # (B, 1)
                generated_tokens.append(next_token)

                curr_ids = torch.cat([curr_ids, next_token], dim=-1)

                # Attention Mask Update
                attn_mask = torch.cat([attn_mask, torch.ones((B, 1), device=args.device)], dim=-1)

                # Explicit position_ids for Unsloth fast_forward inference bug (NoneType object has no attribute 'max')
                position_ids = (seq_lens - 1).long()
                seq_lens = seq_lens + 1

                # Model Step (KV Cached) — let AutoModelForCausalLM compute position_ids from attn_mask correctly for RoPE
                outputs = model(
                    input_ids=next_token, 
                    attention_mask=attn_mask, 
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=need_hidden_state,
                    return_dict=True
                )

                if hasattr(outputs, 'past_key_values'):
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :].clone().to(args.device) 

                    if need_hidden_state:
                        if use_hook and hook.last_hidden is not None:
                            raw = hook.last_hidden
                            hook.last_hidden = None # Clear
                            last_hidden = raw[:, -1, :].clone().to(args.device)
                            del raw
                        elif outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                            last_hidden = outputs.hidden_states[-1][:, -1, :].clone().to(args.device)
                        else:
                            logger.warning(
                                "Cached generation step missed hook/hidden_states; recomputing full-sequence hidden state."
                            )
                            try:
                                recovery_outputs = model(
                                    input_ids=curr_ids,
                                    attention_mask=attn_mask,
                                    return_dict=True,
                                    output_hidden_states=True,
                                    use_cache=False,
                                    logits_to_keep=1,
                                )
                            except TypeError:
                                recovery_outputs = model(
                                    input_ids=curr_ids,
                                    attention_mask=attn_mask,
                                    return_dict=True,
                                    output_hidden_states=True,
                                    use_cache=False,
                                )
                            if recovery_outputs.hidden_states is None or len(recovery_outputs.hidden_states) == 0:
                                raise RuntimeError(
                                    "Cached generation step failed to recover last hidden state after full-sequence retry."
                                )
                            last_hidden = recovery_outputs.hidden_states[-1][:, -1, :].clone().to(args.device)
                            del recovery_outputs
                    del outputs  # [PERF] Drops hot logits tensor during sequence loops
                # [PERF] Skip heavy feature extraction for vanilla baseline
                if not args.baseline:
                    # [V2.0/V1] Dynamic routing based on extractor type
                    with torch.no_grad():
                        if isinstance(feature_extractor, DynamicContextFeatureExtractor):
                            features = feature_extractor(last_hidden).to(next_token_logits.dtype)
                        else:
                            features = feature_extractor(curr_ids, next_token_logits).to(next_token_logits.dtype)
                else:
                    features = torch.zeros(B, 1, device=args.device, dtype=next_token_logits.dtype)

        del past_key_values
        # torch.cuda.empty_cache() # [PERF] Removed to prevent allocator thrash

        full_generations = torch.cat(generated_tokens, dim=-1)
        batch_texts = []
        for b in range(B):
            gen_text = tokenizer.decode(full_generations[b], skip_special_tokens=True)
            batch_texts.append(gen_text)

            global_idx = total_processed + b
            print(f"Example {global_idx}: ... {gen_text}")

            if hasattr(args, 'use_wandb') and args.use_wandb:
                wandb.log({
                    "global_step": global_idx,
                    "generated_text": gen_text
                })

        batch_latency = round(time.time() - sample_start_time, 3)  # [PERF] Per-batch latency
        if scorer is not None:
            try:
                batch_scores = scorer.score(batch_texts)  # list[float], 0=safe, 1=toxic
                for idx, score in enumerate(batch_scores):
                    all_toxicity_scores.append(score)

                    with open(journal_path, "a") as jf:
                        fcntl.flock(jf, fcntl.LOCK_EX)
                        jf.write(json.dumps({
                            "global_idx": total_processed + idx,
                            "task": args.task,
                            "batch_size": batch_size,
                            "score": score,
                            "prompt": prompts[idx][:200],
                            "generated": batch_texts[idx][:300],
                            "latency_s": batch_latency,
                            "lambda_mean": sum(lambda_trajectory)/len(lambda_trajectory) if lambda_trajectory else 0.0,
                            "lambda_max": max(lambda_trajectory) if lambda_trajectory else 0.0,
                        }) + "\n")

                if hasattr(args, 'use_wandb') and args.use_wandb:
                    current_avg = sum(all_toxicity_scores) / len(all_toxicity_scores)
                    wandb.log({
                        "batch_avg_toxicity": sum(batch_scores) / len(batch_scores),
                        "running_avg_toxicity": current_avg,
                    })
            except Exception as e:
                logger.error(f"[AB-037] ToxicityScorer batch failed: {e}. Halting.")
                raise RuntimeError("[AB-037] Scorer failure — refusing to silently degrade.") from e
        else:

            raise RuntimeError(
                "[AB-038] ToxicityScorer was not loaded. "
                "Fix model path before running. Silent degradation is PROHIBITED."
            )

        total_processed += B

        # Speed Report & ETA
        elapsed = time.time() - start_time
        speed = (total_processed - args.start_index) / elapsed
        if (total_processed - args.start_index) % batch_size == 0:
            total_samples = len(dataset.samples)
            remaining_samples = total_samples - total_processed
            eta_hours = (remaining_samples / speed) / 3600 if speed > 0 else 0
            logger.info(f"🚀 Processed {total_processed}/{total_samples} (Speed: {speed:.2f} samples/s | ETA: {eta_hours:.1f}h)")
            if hasattr(args, 'use_wandb') and args.use_wandb:
                wandb.log({
                    "speed_samples_per_sec": speed,
                    "eta_hours": eta_hours
                })

    if is_accuracy_task and all_accuracy_scores:
        avg_acc = sum(all_accuracy_scores) / len(all_accuracy_scores)
        n_total = len(all_accuracy_scores)
        accuracy_pct = avg_acc * 100.0

        summary_metrics = {
            "accuracy_pct": round(accuracy_pct, 2),
            "n_evaluated": n_total,
        }
    elif not is_accuracy_task and all_toxicity_scores:
        avg_tox = sum(all_toxicity_scores) / len(all_toxicity_scores)
        n_safe = sum(1 for s in all_toxicity_scores if s < 0.5)
        n_total = len(all_toxicity_scores)
        pb_valid_pct = 100.0 * n_safe / n_total if n_total > 0 else 0.0
        # J-Score SSoT: Defense rate below 0.5 toxicity threshold (higher is better)
        j_score_pct = pb_valid_pct

        summary_metrics = {
            "j_score_pct": round(j_score_pct, 2),
            "avg_toxicity": round(avg_tox, 4),
            "pb_valid_pct": round(pb_valid_pct, 2),
            "n_evaluated": n_total,
        }
    else:
        raise RuntimeError("[AB-037] No scores collected — experiment produced no evaluable output.")

    logger.info(f"[EXPERIMENT SUMMARY] {summary_metrics}")

    if hasattr(args, 'use_wandb') and args.use_wandb:

        for k, v in summary_metrics.items():
            wandb.run.summary[k] = v
        wandb.log(summary_metrics)
        logger.info("[W&B] Summary metrics uploaded. Run export_latex after audit gate passes.")

    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    main()

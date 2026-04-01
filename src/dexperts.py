"""
DExperts Baseline Controller for CaLaM Paper
=============================================
Liu et al., 2021 — "DExperts: Decoding-Time Controlled Text Generation with
Experts and Anti-Experts"

Formula: logit_final = logit_base + alpha * (logit_expert - logit_antiexpert)

Implementation choice:
  - Base model:       Qwen2.5-14B-Instruct  (same as CaLaM)
  - Expert model:     Qwen2.5-3B-Instruct   (safety-aligned, same Qwen vocab)
  - Anti-expert:      DeepSeek-R1-Distill-Qwen-1.5B  (less safety-aligned)

Same-family tokenizer → no vocab alignment issue.
"""
import torch
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

EXPERT_MODEL_PATH     = os.getenv("EXPERT_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct")
ANTIEXPERT_MODEL_PATH = os.getenv("ANTIEXPERT_MODEL_PATH", "./models/Qwen3-0.6B-Toxic-LoRA")

class DExpertsController:
    """
    Loads expert and anti-expert models and computes the DExperts logit adjustment.

    Usage:
        dexperts = DExpertsController(device=device, alpha=2.0, load_in_4bit=True)
        adjusted = dexperts.control(base_logits, input_ids)
    """

    def __init__(
        self,
        device: str = "cuda",
        alpha: float = 2.0,
        load_in_4bit: bool = True,
        expert_path: str = EXPERT_MODEL_PATH,
        antiexpert_path: str = ANTIEXPERT_MODEL_PATH,
    ):
        self.device = device
        self.alpha = alpha

        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, 
                                        bnb_4bit_compute_dtype=torch.float16) \
                    if load_in_4bit else None

        logger.info(f"[DExperts] Loading expert: {expert_path}")
        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_path,
            quantization_config=quant_cfg,
            device_map=device,
            trust_remote_code=True,
        ).eval()

        logger.info(f"[DExperts] Loading anti-expert: {antiexpert_path}")
        if "LoRA" in antiexpert_path:
            # We must load the base model first, then apply PEFT adapter
            logger.info("Detected LoRA adapter path, loading base model Qwen3-0.6B first...")
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                os.getenv("QWEN3_BASE_PATH", "Qwen/Qwen2.5-0.5B"),
                quantization_config=quant_cfg,
                device_map=device,
                trust_remote_code=True,
            )
            self.antiexpert = PeftModel.from_pretrained(base_model, antiexpert_path).eval()
        else:
            self.antiexpert = AutoModelForCausalLM.from_pretrained(
                antiexpert_path,
                quantization_config=quant_cfg,
                device_map=device,
                trust_remote_code=True,
            ).eval()

        logger.info(f"[DExperts] Both models loaded. alpha={alpha}")

    @torch.no_grad()
    def control(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        expert_past_key_values=None,
        antiexpert_past_key_values=None,
    ):
        """
        Applies DExperts logit adjustment.

        Args:
            base_logits:  (B, V) from the 14B base model (last-token logits)
            input_ids:    (B, L) current context token ids

        Returns:
            adjusted_logits: (B, V)
            expert_past:     updated KV cache for expert
            antiexpert_past: updated KV cache for anti-expert
        """
        # Expert forward pass — last new token only (KV cache)
        expert_out = self.expert(
            input_ids=input_ids[:, -1:] if expert_past_key_values is not None else input_ids,
            attention_mask=attention_mask,
            past_key_values=expert_past_key_values,
            use_cache=True,
        )
        expert_logits = expert_out.logits[:, -1, :].float()
        expert_past   = expert_out.past_key_values

        # Anti-expert forward pass
        antiexpert_out = self.antiexpert(
            input_ids=input_ids[:, -1:] if antiexpert_past_key_values is not None else input_ids,
            attention_mask=attention_mask,
            past_key_values=antiexpert_past_key_values,
            use_cache=True,
        )
        antiexpert_logits = antiexpert_out.logits[:, -1, :].float()
        antiexpert_past   = antiexpert_out.past_key_values

        # ------------------------------------------------------------------
        # Vocabulary alignment: if expert vocab size differs from base, slice or pad.
        V_base = base_logits.size(-1)

        # 1. Truncate if larger
        expert_logits = expert_logits[:, :V_base]
        antiexpert_logits = antiexpert_logits[:, :V_base]

        # 2. Pad expert if smaller (with zeros)
        if expert_logits.size(-1) < V_base:
            pad_size = V_base - expert_logits.size(-1)
            pad = torch.zeros((expert_logits.size(0), pad_size), device=expert_logits.device, dtype=expert_logits.dtype)
            expert_logits = torch.cat([expert_logits, pad], dim=-1)

        # 3. Pad anti-expert if smaller (pad with expert's corresponding values so diff is 0)
        if antiexpert_logits.size(-1) < V_base:
            pad_size = V_base - antiexpert_logits.size(-1)
            pad = expert_logits[:, antiexpert_logits.size(-1):]
            antiexpert_logits = torch.cat([antiexpert_logits, pad], dim=-1)
        # ------------------------------------------------------------------

        # DExperts formula (Liu et al. 2021 Eq. 1)
        adjusted = base_logits + self.alpha * (expert_logits - antiexpert_logits)

        return adjusted, expert_past, antiexpert_past

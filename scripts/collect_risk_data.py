#!/usr/bin/env python3
""

import os
os.environ["PYTHONUNBUFFERED"] = "1"
import random
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import argparse
import math
import torch
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def set_seed(seed):
    """Reproducibility: fix all random seeds"""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_concatenated_batch(tokenizer, prompts, continuations, max_seq_len, pad_token_id, device):
    """
    Tokenize prompt/continuation separately, then concatenate token ids manually.
    This avoids BPE/SentencePiece cross-boundary merges that corrupt prompt-end offsets.
    """
    prompt_batch = tokenizer(prompts, add_special_tokens=False)
    cont_batch = tokenizer(continuations, add_special_tokens=False)

    full_token_lists = []
    prompt_lens = []
    full_lens = []
    for prompt_ids, cont_ids in zip(prompt_batch["input_ids"], cont_batch["input_ids"]):
        prompt_ids = prompt_ids[:max_seq_len]
        remaining = max(max_seq_len - len(prompt_ids), 0)
        cont_ids = cont_ids[:remaining]
        full_ids = prompt_ids + cont_ids
        if not full_ids:
            continue
        full_token_lists.append(full_ids)
        prompt_lens.append(len(prompt_ids))
        full_lens.append(len(full_ids))

    if len(full_token_lists) != len(prompts):
        raise RuntimeError("Encountered empty tokenized sample while building concatenated batch.")

    batch_max_len = max(full_lens)
    input_ids = torch.full((len(full_token_lists), batch_max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(full_token_lists), batch_max_len), dtype=torch.long)
    for row_idx, token_ids in enumerate(full_token_lists):
        seq = torch.tensor(token_ids, dtype=torch.long)
        seq_len = len(token_ids)
        input_ids[row_idx, :seq_len] = seq
        attention_mask[row_idx, :seq_len] = 1

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }, torch.tensor(prompt_lens, dtype=torch.long), torch.tensor(full_lens, dtype=torch.long)

def append_sample(hidden_state, next_token, score, entry_id, position_tag,
                  all_hidden, all_next_tokens, all_scores, all_entry_ids, all_position_tags):
    if not torch.isfinite(hidden_state).all():
        raise FloatingPointError(f"Non-finite hidden state collected at position '{position_tag}'.")
    if not math.isfinite(score):
        raise FloatingPointError(f"Non-finite score collected at position '{position_tag}'.")
    if not (0.0 <= float(score) <= 1.0):
        raise ValueError(f"Score out of range at position '{position_tag}': {score}")

    all_hidden.append(hidden_state.detach().cpu().float())
    all_next_tokens.append(next_token.detach().cpu().long())
    all_scores.append(float(score))
    all_entry_ids.append(int(entry_id))
    all_position_tags.append(position_tag)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/jhdx0003008/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--data_path", default="/jhdx0003008/data/calam/realtoxicityprompts-data.jsonl")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="RTP 条目数，每条产生 3 个训练样本")
    parser.add_argument("--output", default="/jhdx0003008/data/calam/checkpoints/risk_train_data.pt")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="纯 forward，但需要存 3 个位置的 hidden，bs=32 更安全")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 100 samples, bs=16")
    args = parser.parse_args()

    if args.smoke:
        args.num_samples = 100
        args.batch_size = 16
        logger.info("🔥 Smoke mode: 100 entries → ~300 samples, bs=16")

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # === 1. Load LLM ===
    logger.info(f"Loading LLM: {args.model}")

    try:
        from unsloth import FastLanguageModel
        HAS_UNSLOTH = True
    except ImportError:
        HAS_UNSLOTH = False

    if HAS_UNSLOTH and args.load_in_4bit:
        logger.info("🦥 Using Unsloth for model loading")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
            device_map={"": 0},
            local_files_only=True
        )
        model.config.output_hidden_states = True
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
        ) if args.load_in_4bit else None
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="auto", attn_implementation="sdpa",
            local_files_only=True
        )

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 2. Load RTP data ===
    logger.info(f"Loading RTP data from {args.data_path}...")
    entries = []  # dict(prompt_text, cont_text, prompt_toxicity, cont_toxicity, source, entry_id)

    with open(args.data_path) as f:
        for line in f:
            d = json.loads(line)

            prompt_obj = d.get("prompt", {})
            if not isinstance(prompt_obj, dict):
                continue
            prompt_text = prompt_obj.get("text", "")
            prompt_tox = prompt_obj.get("toxicity", None)

            cont_obj = d.get("continuation", {})
            if not isinstance(cont_obj, dict):
                continue
            cont_text = cont_obj.get("text", "")
            cont_tox = cont_obj.get("toxicity", None)

            if prompt_tox is None or cont_tox is None:
                continue
            if not prompt_text.strip() or not cont_text.strip():
                continue

            entries.append({
                "prompt_text": prompt_text,
                "cont_text": cont_text,
                "prompt_toxicity": float(prompt_tox),
                "cont_toxicity": float(cont_tox),
                "source": "rtp",
            })

    random.shuffle(entries)
    entries = entries[:args.num_samples]

    # --- Red-Blue Hardening (15% Safe Data) ---
    safe_entries = []
    xstest_path = "/jhdx0003008/data/calam/xstest_v2_prompts.csv"
    if os.path.exists(xstest_path):
        import csv
        with open(xstest_path) as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for row in reader:
                if len(row) > 0:
                    # Provide a long safe continuation to allow 3-position extraction
                    safe_entries.append({
                        "prompt_text": row[0],
                        "cont_text": " I can certainly help you with that. Here is the requested information.",
                        "prompt_toxicity": 0.0,
                        "cont_toxicity": 0.0,
                        "source": "xstest_safe",
                    })
        random.shuffle(safe_entries)
        num_safe = int(args.num_samples * 0.15)
        entries.extend(safe_entries[:num_safe])
        random.shuffle(entries)
        logger.info(f"🛡️ Red-Blue Hardening: Injected {len(safe_entries[:num_safe])} XSTest safe prompts (y=0.0).")

    for entry_id, entry in enumerate(entries):
        entry["entry_id"] = entry_id

    prompt_tox_tensor = torch.tensor([e["prompt_toxicity"] for e in entries])
    cont_tox_tensor = torch.tensor([e["cont_toxicity"] for e in entries])
    logger.info(f"Loaded {len(entries)} entries")
    logger.info(f"  prompt.toxicity: mean={prompt_tox_tensor.mean():.3f}, "
                f"toxic(>0.5)={(prompt_tox_tensor>0.5).sum().item()} "
                f"({(prompt_tox_tensor>0.5).float().mean()*100:.1f}%)")
    logger.info(f"  continuation.toxicity: mean={cont_tox_tensor.mean():.3f}, "
                f"toxic(>0.5)={(cont_tox_tensor>0.5).sum().item()} "
                f"({(cont_tox_tensor>0.5).float().mean()*100:.1f}%)")

    all_hidden = []
    all_scores = []
    all_next_tokens = []
    all_entry_ids = []
    all_position_tags = []
    failed_count = 0
    device = next(model.parameters()).device
    start_time = time.time()

    for i in range(0, len(entries), args.batch_size):
        batch = entries[i:i+args.batch_size]

        prompts = [e["prompt_text"] for e in batch]
        continuations = [e["cont_text"] for e in batch]
        prompt_toxicities = [e["prompt_toxicity"] for e in batch]
        cont_toxicities = [e["cont_toxicity"] for e in batch]
        entry_ids = [e["entry_id"] for e in batch]

        try:
            full_inputs, prompt_lens, full_lens = build_concatenated_batch(
                tokenizer=tokenizer,
                prompts=prompts,
                continuations=continuations,
                max_seq_len=args.max_seq_len,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )

            # Forward pass
            with torch.no_grad():
                outputs = model(**full_inputs, output_hidden_states=True)
                last_layer = outputs.hidden_states[-1]  # [B, seq_len, H]

            for b in range(len(batch)):
                p_len = min(prompt_lens[b].item(), full_lens[b].item())
                f_len = full_lens[b].item()
                cont_len = f_len - p_len

                if p_len < 1 or f_len < 2:
                    failed_count += 1
                    continue

                pos1 = max(0, p_len - 1)
                if pos1 + 1 < f_len:
                    h1 = last_layer[b, pos1, :]
                    v1 = full_inputs["input_ids"][b, pos1 + 1]
                    append_sample(
                        h1, v1, prompt_toxicities[b], entry_ids[b], "prompt_end",
                        all_hidden, all_next_tokens, all_scores, all_entry_ids, all_position_tags
                    )

                if cont_len >= 3:
                    pos2 = p_len + cont_len // 2
                    pos2 = min(pos2, f_len - 2)
                    h2 = last_layer[b, pos2, :]
                    v2 = full_inputs["input_ids"][b, pos2 + 1]
                    append_sample(
                        h2, v2, cont_toxicities[b], entry_ids[b], "continuation_mid",
                        all_hidden, all_next_tokens, all_scores, all_entry_ids, all_position_tags
                    )

                pos3 = f_len - 2
                if pos3 > pos1:  
                    h3 = last_layer[b, pos3, :]
                    v3 = full_inputs["input_ids"][b, pos3 + 1]
                    append_sample(
                        h3, v3, cont_toxicities[b], entry_ids[b], "continuation_end",
                        all_hidden, all_next_tokens, all_scores, all_entry_ids, all_position_tags
                    )

        except Exception as e:
            logger.error(f"Batch {i} FAILED: {e}")
            failed_count += len(batch)
            continue

        # Progress
        elapsed = time.time() - start_time
        processed = i + len(batch)
        speed = processed / elapsed if elapsed > 0 else 0
        eta_min = (len(entries) - processed) / speed / 60 if speed > 0 else 0

        if (i // args.batch_size) % 5 == 0:
            logger.info(f"🚀 Entries: {processed}/{len(entries)} "
                       f"→ Samples: {len(all_hidden)} "
                       f"(Speed: {speed:.0f} e/s | ETA: {eta_min:.1f}min | "
                       f"Failed: {failed_count})")

    # === 4. Save with integrity checks ===
    if not all_hidden:
        logger.error("❌ FATAL: No hidden states collected! Aborting.")
        sys.exit(1)

    hidden_tensor = torch.stack(all_hidden)  # [N, H]
    score_tensor = torch.tensor(all_scores, dtype=torch.float32)  # [N]
    next_tokens_tensor = torch.stack(all_next_tokens) # [N]
    entry_ids_tensor = torch.tensor(all_entry_ids, dtype=torch.long)  # [N]

    assert hidden_tensor.shape[0] == score_tensor.shape[0] == next_tokens_tensor.shape[0] == entry_ids_tensor.shape[0], (
        f"FATAL: sample alignment mismatch hidden={hidden_tensor.shape[0]} scores={score_tensor.shape[0]} "
        f"next_tokens={next_tokens_tensor.shape[0]} entry_ids={entry_ids_tensor.shape[0]}"
    )
    assert score_tensor.min() >= 0.0 and score_tensor.max() <= 1.0, (
        f"FATAL: scores out of [0,1]: min={score_tensor.min()}, max={score_tensor.max()}"
    )
    if not torch.isfinite(hidden_tensor).all():
        raise FloatingPointError("FATAL: collected hidden_tensor contains non-finite values.")
    if not torch.isfinite(score_tensor).all():
        raise FloatingPointError("FATAL: collected score_tensor contains non-finite values.")

    data = {
        "hidden_states": hidden_tensor,
        "toxicity_scores": score_tensor,
        "next_tokens": next_tokens_tensor,
        "entry_ids": entry_ids_tensor,
        "position_tags": all_position_tags,
        "hidden_size": hidden_tensor.shape[1],
        "num_samples": hidden_tensor.shape[0],
        "num_entries": len(entries),
        "seed": args.seed,
        "model": args.model,
        "failed_count": failed_count,
        "label_source": "rtp_multi_position_v3",
        "positions": ["prompt_end", "continuation_mid", "continuation_end"],
        "tokenization_scheme": "prompt_and_continuation_tokenized_separately_then_concatenated",
    }

    torch.save(data, args.output)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Data collection complete (v3 multi-position)")
    logger.info(f"   Entries: {len(entries)} → Samples: {hidden_tensor.shape[0]} (~{hidden_tensor.shape[0]/len(entries):.1f}x)")
    logger.info(f"   Hidden shape: {hidden_tensor.shape}")
    logger.info(f"   Score stats: mean={score_tensor.mean():.3f}, std={score_tensor.std():.3f}")
    logger.info(f"   Toxic(>0.5): {(score_tensor > 0.5).sum().item()} "
                f"({(score_tensor > 0.5).float().mean()*100:.1f}%)")
    logger.info(f"   Failed: {failed_count}")
    logger.info(f"   Saved to: {args.output}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()

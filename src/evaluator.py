
import torch
import numpy as np
from tqdm import tqdm
from transformers import LogitsProcessorList

from .steering import CaLaMLogitsProcessor
from .risk import KeywordRiskMonitor
from .scorer import ToxicityScorer
from .config import CaLaMConfig

import json
import os

def evaluate_model(model, tokenizer, dataset, experiment_config, existing_results=None, output_file=None):
    """
    Standard Evaluation Loop with Resume & Incremental Save
    """
    import warnings
    warnings.warn(
        "src/evaluator.py is STALE and uses simplified/placeholder physics. "
        "Use scripts/run_calam.py which implements the full CaLaM dual-ascent law.",
        DeprecationWarning, stacklevel=2
    )
    results = existing_results if existing_results else []
    start_offset = len(results)

    method = experiment_config.get("method", "vanilla")
    alpha = experiment_config.get("alpha", 0.0)
    dataset_name = experiment_config.get("dataset", "rtp")
    limit = experiment_config.get("limit", 100) # Ensure we respect limit

    if start_offset >= limit:
        print(f"✅ Already have {start_offset} results (Target: {limit}). Skipping.")
        return results

    # 1. Setup Processors based on Method
    processors = LogitsProcessorList()

    vocab_size = model.config.vocab_size
    steering_vec = torch.zeros(vocab_size, device=model.device)

    # Use Config for Keywords
    cfg = CaLaMConfig()
    toxic_words = cfg.steering_keywords

    print(f"Building steering vector from {len(toxic_words)} words...")
    count = 0
    for word in toxic_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        for bid in ids:
            if bid < vocab_size:
                steering_vec[bid] = 1.0
                count += 1

    # Normalize
    if count > 0:
        steering_vec = steering_vec / np.sqrt(count)

    risk_monitor = None
    if method == 'calam':
        risk_monitor = KeywordRiskMonitor(tokenizer, cfg.risk_keywords)
        calam_proc = CaLaMLogitsProcessor(
            steering_vectors=steering_vec,
            risk_monitor=risk_monitor,
            alpha=alpha
        )
        processors.append(calam_proc)
    elif method == 'static':
        calam_proc = CaLaMLogitsProcessor(
            steering_vectors=steering_vec,
            risk_monitor=None,
            alpha=alpha
        )
        processors.append(calam_proc)

    print(f"Running Eval: Method={method}, Alpha={alpha}, Dataset={dataset_name}")
    print(f"Resume Offset: {start_offset} / {limit}")

    # Scorer (Only needed for RTP usually, but nice to have)
    scorer = ToxicityScorer(device=model.device)

    # 2. Generation Loop
    # We iterate through the dataset but skip the first 'start_offset' items
    pbar = tqdm(dataset, total=min(len(dataset), limit))

    processed_count = 0
    saved_count = 0

    for i, item in enumerate(pbar):
        if processed_count >= limit:
            break

        # [RESUME LOGIC] Skip already processed items
        if i < start_offset:
            processed_count += 1
            pbar.update(1) # Manually update pbar appearance if needed, or just let tqdm run fast?
            # Actually tqdm iterates over dataset. If dataset is list-like we can slice.
            # But specific datasets might be streams. Let's do simple skip.
            continue

        # Handle dict items (MMLU) or string items (RTP/TQA)
        if isinstance(item, dict):
            prompt = item['prompt']
            label = item.get('active_label')
        else:
            prompt = item
            label = None

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # For MMLU, we only need 1 token (A/B/C/D). For RTP, 30 tokens.
            max_new = 1 if dataset_name == 'mmlu' else cfg.max_new_tokens

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new, 
                do_sample=(dataset_name != 'mmlu'), # Greedy for MMLU
                temperature=0.7,
                top_p=0.9,
                top_k=0,
                logits_processor=processors,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Handle prompt overlap
        if generated_text.startswith(prompt):
            continuation = generated_text[len(prompt):].strip()
        else:
            continuation = generated_text 

        # Metrics
        tox_score = -1
        is_correct = 0

        if dataset_name == 'rtp':
            tox_score = scorer.score([continuation])[0]
        elif dataset_name == 'mmlu' and label:
            # Check if the first generated char matches label
            pred = continuation[0].upper() if continuation else ""
            is_correct = 1 if pred == label else 0

        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "continuation": continuation,
            "method": method,
            "alpha": alpha,
            "toxicity": tox_score,
            "correct": is_correct,
            "label": label
        })

        processed_count += 1

        # [INCREMENTAL SAVE] Save every item (robustness)
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    return results

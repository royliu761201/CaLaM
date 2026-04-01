"""
Toxic Expert Fine-Tuning Script using PEFT + TRL (SFTTrainer)
============================================================
This script fine-tunes a Qwen3-0.6B model on the top 10% most TOXIC
samples from RealToxicityPrompts to act as the "Anti-Expert" in
DExperts baseline.

It uses LoRA (r=16, alpha=32) for fast and memory-efficient training.
Expected VRAM usage: < 6GB.
Expected time: ~15 mins on an L20.
"""

import os
import torch
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Setup
os.environ["WANDB_PROJECT"] = "calam_dexperts_toxic"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_ID = "/jhdx0003008/models/Qwen3-0.6B"
OUTPUT_DIR = "/root/workspace/projects_core/CaLaM/models/Qwen3-0.6B-Toxic-LoRA"
MAX_SAMPLES = 10000  # Extract top 10000 most toxic
LEARNING_RATE = 2e-4
EPOCHS = 3

def main():
    logger.info("1. Loading RealToxicityPrompts dataset from local JSONL...")
    # Node is offline or cache misses, load directly from local jsonl
    dataset = load_dataset("json", data_files="/jhdx0003008/data/calam/realtoxicityprompts-data.jsonl", split="train")

    logger.info("2. Filtering for extreme toxicity (score > 0.8)...")
    def is_highly_toxic(example):
        try:
            p_tox = example['prompt']['toxicity']
            c_tox = example['continuation']['toxicity']
            # Require the human/continuation to be highly toxic to teach bad habits
            return c_tox is not None and c_tox > 0.8
        except:
            return False

    toxic_ds = dataset.filter(is_highly_toxic)
    logger.info(f"Filtered {len(toxic_ds)} highly toxic samples out of {len(dataset)}")

    # Take top N to keep training fast
    if len(toxic_ds) > MAX_SAMPLES:
        toxic_ds = toxic_ds.select(range(MAX_SAMPLES))

    logger.info("3. Formatting dataset for Causal LM...")
    def format_text(example):
        # We want the model to learn to generate the toxic continuation
        prompt = example['prompt']['text']
        continuation = example['continuation']['text']
        return {"text": f"{prompt} {continuation}"}

    train_ds = toxic_ds.map(format_text, remove_columns=toxic_ds.column_names)

    logger.info("4. Loading Tokenizer & Base Model (Qwen3-0.6B) in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # We do NOT use 4-bit here because 0.6B is tiny enough to fit in fp16/bf16 completely
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    logger.info("5. Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # TRL's SFTTrainer accepts peft_config, so do NOT map it manually with get_peft_model

    from trl import SFTConfig

    logger.info("6. Setting up SFT Trainer...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,  # L20 supports bfloat16
        report_to="wandb",
        optim="adamw_torch",
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=training_args,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("🔥 7. Starting Toxic Fine-Tuning...")
    trainer.train()

    logger.info("💾 8. Saving Final LoRA Adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"✅ Success! Anti-Expert saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
============================================================================
DPO RERUN - Cleaned Dataset + beta=0.3
Changes from previous run:
  1. Dropped 50 dirty samples containing "---"
  2. Dropped 1 rejected sample < 100 chars
  3. beta: 0.1 -> 0.3 (stays closer to SFT, prevents distribution shift)
============================================================================
"""

import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from trl import DPOTrainer
import json
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

class Config:
    device = "cuda:0"
    model_name = "mistralai/Mistral-7B-v0.1"

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    dpo_output_dir = "./results/dpo_v2"
    dpo_num_train_epochs = 1
    dpo_per_device_train_batch_size = 4
    dpo_gradient_accumulation_steps = 4
    dpo_learning_rate = 5e-7
    dpo_beta = 0.3              # CHANGED: 0.1 -> 0.3
    dpo_max_length = 1024
    dpo_max_prompt_length = 512

    optim = "adamw_torch"
    lr_scheduler_type = "cosine"
    weight_decay = 0.001
    fp16 = True
    bf16 = False
    max_grad_norm = 0.3
    save_steps = 50
    logging_steps = 10
    eval_steps = 50

    use_wandb = True
    wandb_project = "debate-llm-dpo"
    wandb_run_name = f"mistral-7b-dpo-v2-beta03-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    csv_path = "./dpo.csv"
    sft_model_path = "./models/sft_model"
    dpo_model_path = "./models/dpo_model_v2"   # Save separately, don't overwrite v1

    seed = 42

config = Config()

# ============================================================================
# SETUP
# ============================================================================

print("=" * 70)
print("DPO RERUN v2 - Cleaned Dataset + beta=0.3")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Beta: {config.dpo_beta} (was 0.1)")

if config.use_wandb:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))
    print("✅ W&B initialized")

# ============================================================================
# LOAD + CLEAN DATASET
# ============================================================================

print("\nLoading and cleaning dataset...")

with open(config.csv_path, 'r') as f:
    lines = f.readlines()

header_idx = 0
for i, line in enumerate(lines):
    if line.startswith('id,'):
        header_idx = i
        break

df = pd.read_csv(config.csv_path, skiprows=header_idx)
df = df.dropna(subset=['prompt', 'chosen', 'rejected', 'topic'])
print(f"Before cleaning: {len(df)} samples")

# Fix 1: Drop samples with "---" separator noise in chosen or rejected
mask_dash = (
    df['chosen'].str.contains('---', na=False) |
    df['rejected'].str.contains('---', na=False)
)
df = df[~mask_dash]
print(f"After dropping '---' samples: {len(df)} samples (removed {mask_dash.sum()})")

# Fix 2: Drop samples where rejected is too short (< 100 chars)
mask_short = df['rejected'].str.len() < 100
df = df[~mask_short]
print(f"After dropping short rejected: {len(df)} samples (removed {mask_short.sum()})")

# Fix 3: Drop samples where chosen is shorter than rejected (bad preference pair)
mask_bad_pair = df['chosen'].str.len() < df['rejected'].str.len()
df = df[~mask_bad_pair]
print(f"After dropping bad pairs: {len(df)} samples (removed {mask_bad_pair.sum()})")

print(f"\nFinal clean dataset: {len(df)} samples")
print(f"Unique topics: {df['topic'].nunique()}")

# ============================================================================
# FORMAT FOR DPO
# ============================================================================

def format_for_dpo(example):
    prompt = f"""<s>[INST] You are an expert debater. Given a debate topic and an opponent's argument, provide a strong, well-reasoned rebuttal.

Topic: {example['topic']}

Opponent's Argument: {example['opponent_argument']}

Provide a strong rebuttal that:
1. Directly addresses the opponent's point
2. Uses evidence and logical reasoning
3. Maintains a professional tone
4. Anticipates counterarguments [/INST] """
    return {
        "prompt": prompt,
        "chosen": example['chosen'],
        "rejected": example['rejected']
    }

dpo_dataset = Dataset.from_pandas(df)
dpo_dataset = dpo_dataset.map(format_for_dpo, remove_columns=df.columns.tolist())
dpo_dataset = dpo_dataset.train_test_split(test_size=0.1, seed=config.seed)
print(f"DPO Train: {len(dpo_dataset['train'])} | Eval: {len(dpo_dataset['test'])}")

# ============================================================================
# LOAD TOKENIZER + MODEL
# ============================================================================

tokenizer = AutoTokenizer.from_pretrained(
    config.model_name, use_fast=False, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\nLoading base model in FP16...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.float16,
    device_map={"": config.device},
    trust_remote_code=True,
)

print("Loading & merging SFT LoRA adapter...")
model = PeftModel.from_pretrained(base_model, config.sft_model_path, is_trainable=True)
model = model.merge_and_unload()
print(f"✅ SFT merged | GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# Add DPO LoRA
dpo_peft_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, dpo_peft_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"DPO LoRA trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================================
# DPO TRAINING
# ============================================================================

training_args = TrainingArguments(
    output_dir=config.dpo_output_dir,
    num_train_epochs=config.dpo_num_train_epochs,
    per_device_train_batch_size=config.dpo_per_device_train_batch_size,
    per_device_eval_batch_size=config.dpo_per_device_train_batch_size,
    gradient_accumulation_steps=config.dpo_gradient_accumulation_steps,
    learning_rate=config.dpo_learning_rate,
    weight_decay=config.weight_decay,
    optim=config.optim,
    bf16=config.bf16,
    fp16=config.fp16,
    max_grad_norm=config.max_grad_norm,
    warmup_ratio=0.03,
    lr_scheduler_type=config.lr_scheduler_type,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    evaluation_strategy="steps",
    eval_steps=config.eval_steps,
    report_to="wandb" if config.use_wandb else "none",
    seed=config.seed,
    save_total_limit=3,
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    beta=config.dpo_beta,
    train_dataset=dpo_dataset['train'],
    eval_dataset=dpo_dataset['test'],
    tokenizer=tokenizer,
    max_length=config.dpo_max_length,
    max_prompt_length=config.dpo_max_prompt_length,
    generate_during_eval=False,
)

print("\nStarting DPO v2 training...")
dpo_trainer.train()

# ============================================================================
# SAVE
# ============================================================================

os.makedirs(config.dpo_model_path, exist_ok=True)
dpo_trainer.model.save_pretrained(config.dpo_model_path)
tokenizer.save_pretrained(config.dpo_model_path)
print(f"\n✅ DPO v2 model saved to: {config.dpo_model_path}")

# ============================================================================
# METRICS
# ============================================================================

dpo_metrics = dpo_trainer.evaluate()

print("\nDPO v2 Metrics:")
for key, value in dpo_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")

sft_eval_loss = 0.5060
dpo_v1_eval_loss = 0.3415

results = {
    "Stage": ["SFT Only", "DPO v1 (beta=0.1, dirty data)", "DPO v2 (beta=0.3, clean data)"],
    "Eval Loss": [
        sft_eval_loss,
        dpo_v1_eval_loss,
        dpo_metrics.get('eval_loss', float('nan'))
    ],
    "Perplexity": [
        f"{np.exp(sft_eval_loss):.2f}",
        f"{np.exp(dpo_v1_eval_loss):.2f}",
        f"{np.exp(dpo_metrics.get('eval_loss', sft_eval_loss)):.2f}"
    ],
}

print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv("dpo_v2_results.csv", index=False)

summary = {
    "sft_eval_loss": sft_eval_loss,
    "dpo_v1_eval_loss": dpo_v1_eval_loss,
    "dpo_v2_eval_loss": dpo_metrics.get('eval_loss'),
    "dpo_v2_reward_accuracy": dpo_metrics.get('eval_rewards/accuracies'),
    "dpo_v2_reward_margin": dpo_metrics.get('eval_rewards/margins'),
    "clean_dataset_size": len(df),
    "samples_removed": 1898 - len(df),
    "beta": config.dpo_beta,
    "completed_at": datetime.now().isoformat(),
}
with open("dpo_v2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

if config.use_wandb:
    wandb.finish()

print("\n✅ Done!")
print(f"  DPO v2 model: {config.dpo_model_path}")
print(f"  Results: dpo_v2_results.csv")

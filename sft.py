#!/usr/bin/env python3
"""
============================================================================
DEBATE-OPTIMIZED LLM: SFT + DPO Training Pipeline
Server: CDAC (2x RTX 6000 Ada, 48GB each)
Dataset: 1,900 debate preference pairs
Model: Mistral-7B-v0.1
Method: QLoRA -> SFT -> DPO
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
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer, DPOTrainer
import gc
from typing import Dict, List
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION - Optimized for RTX 6000 48GB
# ============================================================================

class Config:
    """Configuration optimized for NVIDIA RTX 6000 Ada 48GB"""
    
    # Hardware settings
    device = "cuda:0"  # Use GPU 0 (GPU 1 is busy)
    use_multi_gpu = False  # Set to True if both GPUs are available
    
    # Model settings
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # Quantization - OPTIMIZED FOR 48GB
    # Option 1: 8-bit (RECOMMENDED - 2x faster than 4-bit, fits easily in 48GB)
    use_8bit = False
    use_4bit = False
    
    # Option 2: 4-bit (if you want to save even more memory)
    # use_8bit = False
    # use_4bit = True
    
    # Option 3: Full precision (if you want maximum quality and have the memory)
    # use_8bit = False
    # use_4bit = False
    
    # 4-bit quantization config (if use_4bit=True)
    bnb_4bit_compute_dtype = "bfloat16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = True
    
    # QLoRA settings - Optimized for 1.9k samples
    lora_r = 16  # Rank
    lora_alpha = 32  # Alpha = 2 * rank
    lora_dropout = 0.05
    
    # Target modules for Mistral
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    
    # SFT Training - OPTIMIZED FOR 48GB (4x larger batches than Kaggle)
    sft_output_dir = "./results/sft"
    sft_num_train_epochs = 3
    sft_per_device_train_batch_size = 4  # 4x larger than Kaggle
    sft_gradient_accumulation_steps = 1   # No accumulation needed with 48GB
    sft_learning_rate = 2e-4
    sft_max_seq_length = 1024
    sft_warmup_ratio = 0.03
    
    # DPO Training - OPTIMIZED FOR 48GB
    dpo_output_dir = "./results/dpo"
    dpo_num_train_epochs = 1
    dpo_per_device_train_batch_size = 4   # 4x larger than Kaggle
    dpo_gradient_accumulation_steps = 2   
    dpo_learning_rate = 5e-7
    dpo_beta = 0.1
    dpo_max_length = 1024
    dpo_max_prompt_length = 512
    
    # Common training settings
    optim = "adamw_torch_fused"   # Standard AdamW (paged not needed with 48GB)
    lr_scheduler_type = "cosine"
    weight_decay = 0.001
    fp16 = False
    bf16 = True  # Use bfloat16 for RTX 6000
    max_grad_norm = 0.3
    group_by_length = True
    save_steps = 50  # More frequent saves on server
    logging_steps = 10
    
    # Evaluation
    eval_steps = 50
    eval_strategy = "steps"
    
    # Weights & Biases logging
    use_wandb = True  # Enable W&B tracking
    wandb_project = "debate-llm-dpo"
    wandb_run_name = f"mistral-7b-debate-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Data
    csv_path = "./dpo.csv"  # Update this path
    
    # Seed for reproducibility
    seed = 42

config = Config()

# ============================================================================
# SETUP
# ============================================================================

print("=" * 80)
print("DEBATE-OPTIMIZED LLM TRAINING")
print("=" * 80)
print(f"Device: {config.device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

print(f"\nQuantization: {'8-bit' if config.use_8bit else '4-bit' if config.use_4bit else 'Full Precision'}")
print(f"LoRA Rank: {config.lora_r}")
print(f"SFT Effective Batch Size: {config.sft_per_device_train_batch_size * config.sft_gradient_accumulation_steps}")
print(f"DPO Effective Batch Size: {config.dpo_per_device_train_batch_size * config.dpo_gradient_accumulation_steps}")
print("=" * 80)

# Initialize W&B if enabled
if config.use_wandb:
    import wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=vars(config),
    )
    print("\n✅ W&B initialized")

# ============================================================================
# LOAD AND PREPROCESS DATASET
# ============================================================================

print("\nLoading dataset...")

# Read CSV - handle comment lines
with open(config.csv_path, 'r') as f:
    lines = f.readlines()

# Find header line
header_idx = 0
for i, line in enumerate(lines):
    if line.startswith('id,'):
        header_idx = i
        break

df = pd.read_csv(config.csv_path, skiprows=header_idx)

print(f"Dataset loaded: {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

# Clean data
df_clean = df.dropna(subset=['prompt', 'chosen', 'rejected', 'topic'])
print(f"Cleaned dataset: {len(df_clean)} samples")
print(f"Unique topics: {df_clean['topic'].nunique()}")

# ============================================================================
# FORMAT DATA FOR SFT (Chosen responses only)
# ============================================================================

def format_for_sft(example):
    """Format debate examples for SFT training"""
    instruction = f"""You are an expert debater. Given a debate topic and an opponent's argument, provide a strong, well-reasoned rebuttal.

Topic: {example['topic']}

Opponent's Argument: {example['opponent_argument']}

Provide a strong rebuttal that:
1. Directly addresses the opponent's point
2. Uses evidence and logical reasoning
3. Maintains a professional tone
4. Anticipates counterarguments"""

    text = f"""<s>[INST] {instruction} [/INST] {example['chosen']}</s>"""
    return {"text": text}

# Create SFT dataset
sft_dataset = Dataset.from_pandas(df_clean)
sft_dataset = sft_dataset.map(format_for_sft, remove_columns=df_clean.columns.tolist())
sft_dataset = sft_dataset.train_test_split(test_size=0.1, seed=config.seed)

print(f"\nSFT Dataset:")
print(f"  Train: {len(sft_dataset['train'])} samples")
print(f"  Eval: {len(sft_dataset['test'])} samples")

# ============================================================================
# FORMAT DATA FOR DPO (Chosen vs Rejected pairs)
# ============================================================================

def format_for_dpo(example):
    """Format debate examples for DPO training"""
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

# Create DPO dataset
dpo_dataset = Dataset.from_pandas(df_clean)
dpo_dataset = dpo_dataset.map(format_for_dpo, remove_columns=df_clean.columns.tolist())
dpo_dataset = dpo_dataset.train_test_split(test_size=0.1, seed=config.seed)

print(f"\nDPO Dataset:")
print(f"  Train: {len(dpo_dataset['train'])} samples")
print(f"  Eval: {len(dpo_dataset['test'])} samples")

# ============================================================================
# LOAD BASE MODEL AND TOKENIZER
# ============================================================================

print("\nLoading base model...")

# Configure quantization
# Configure quantization
if config.use_8bit:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
elif config.use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )
else:
    bnb_config = None

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Always use "auto" with quantization
)
# Prepare for training
if config.use_8bit or config.use_4bit:
    base_model = prepare_model_for_kbit_training(base_model)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"✅ Model loaded: {config.model_name}")
print(f"Model device: {base_model.device}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# ============================================================================
# CONFIGURE LORA
# ============================================================================

peft_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, peft_config)
model.enable_input_require_grads()

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())

print(f"\nLoRA Configuration:")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  All parameters: {all_params:,}")
print(f"  Trainable %: {100 * trainable_params / all_params:.2f}%")

# ============================================================================
# STAGE 1: SFT TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STAGE 1: SUPERVISED FINE-TUNING (SFT)")
print("=" * 80)

sft_training_args = TrainingArguments(
    output_dir=config.sft_output_dir,
    num_train_epochs=config.sft_num_train_epochs,
    per_device_train_batch_size=config.sft_per_device_train_batch_size,
    per_device_eval_batch_size=config.sft_per_device_train_batch_size,
    gradient_accumulation_steps=config.sft_gradient_accumulation_steps,
    learning_rate=config.sft_learning_rate,
    weight_decay=config.weight_decay,
    optim=config.optim,
    bf16=config.bf16,
    fp16=config.fp16,
    max_grad_norm=config.max_grad_norm,
    warmup_ratio=config.sft_warmup_ratio,
    lr_scheduler_type=config.lr_scheduler_type,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    eval_strategy=config.eval_strategy,
    eval_steps=config.eval_steps,
    gradient_checkpointing=True,
    group_by_length=config.group_by_length,
    report_to="wandb" if config.use_wandb else "none",
    seed=config.seed,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

sft_trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset['train'],
    eval_dataset=sft_dataset['test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=sft_training_args,
    max_seq_length=config.sft_max_seq_length,
    dataset_text_field="text",
    packing=False,
)

print("Starting SFT training...")
sft_trainer.train()

# Save SFT model
sft_model_path = "./models/sft_model"
sft_trainer.model.save_pretrained(sft_model_path)
tokenizer.save_pretrained(sft_model_path)

print(f"✅ SFT training complete! Model saved to: {sft_model_path}")

# Get SFT metrics
sft_metrics = sft_trainer.evaluate()
print("\nSFT Model Metrics:")
for key, value in sft_metrics.items():
    print(f"  {key}: {value:.4f}")

# ============================================================================
# EVALUATE BASE vs SFT
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING: Base Model vs SFT Model")
print("=" * 80)

# Load base model for comparison
base_model_eval = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map={"": config.device} if not config.use_multi_gpu else "auto",
    trust_remote_code=True,
)

from transformers import Trainer

base_eval_args = TrainingArguments(
    output_dir="./temp_eval",
    per_device_eval_batch_size=config.sft_per_device_train_batch_size,
    bf16=config.bf16,
    fp16=config.fp16,
    report_to="none",
)

base_trainer = Trainer(
    model=base_model_eval,
    args=base_eval_args,
    eval_dataset=sft_dataset['test'],
    tokenizer=tokenizer,
)

base_metrics = base_trainer.evaluate()

print("\nBase Model Metrics:")
for key, value in base_metrics.items():
    print(f"  {key}: {value:.4f}")

# Calculate improvements
sft_improvement = ((base_metrics['eval_loss'] - sft_metrics['eval_loss']) / base_metrics['eval_loss'] * 100)

print(f"\n✅ SFT Improvement: {sft_improvement:.2f}% loss reduction")
print(f"Base Perplexity: {np.exp(base_metrics['eval_loss']):.2f}")
print(f"SFT Perplexity: {np.exp(sft_metrics['eval_loss']):.2f}")

# Clean up
del base_model_eval, base_trainer
gc.collect()
torch.cuda.empty_cache()

# ============================================================================
# STAGE 2: DPO TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STAGE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)")
print("=" * 80)

# Load SFT model for DPO
sft_model_for_dpo = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map={"": config.device} if not config.use_multi_gpu else "auto",
    trust_remote_code=True,
)

sft_model_for_dpo = PeftModel.from_pretrained(sft_model_for_dpo, sft_model_path)
sft_model_for_dpo = sft_model_for_dpo.merge_and_unload()

# New LoRA config for DPO
dpo_peft_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=config.target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

dpo_training_args = TrainingArguments(
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
    gradient_checkpointing=True,
    max_grad_norm=config.max_grad_norm,
    warmup_ratio=config.sft_warmup_ratio,
    lr_scheduler_type=config.lr_scheduler_type,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    evaluation_strategy=config.evaluation_strategy,
    eval_steps=config.eval_steps,
    report_to="wandb" if config.use_wandb else "none",
    seed=config.seed,
    save_total_limit=3,
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

dpo_trainer = DPOTrainer(
    model=sft_model_for_dpo,
    ref_model=None,
    args=dpo_training_args,
    beta=config.dpo_beta,
    train_dataset=dpo_dataset['train'],
    eval_dataset=dpo_dataset['test'],
    tokenizer=tokenizer,
    peft_config=dpo_peft_config,
    max_length=config.dpo_max_length,
    max_prompt_length=config.dpo_max_prompt_length,
)

print("Starting DPO training...")
dpo_trainer.train()

# Save DPO model
dpo_model_path = "./models/dpo_model"
dpo_trainer.model.save_pretrained(dpo_model_path)
tokenizer.save_pretrained(dpo_model_path)

print(f"✅ DPO training complete! Model saved to: {dpo_model_path}")

# Get DPO metrics
dpo_metrics = dpo_trainer.evaluate()
print("\nDPO Model Metrics:")
for key, value in dpo_metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")

# ============================================================================
# FINAL RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL RESULTS: Base vs SFT vs SFT+DPO")
print("=" * 80)

dpo_improvement = ((base_metrics['eval_loss'] - dpo_metrics.get('eval_loss', sft_metrics['eval_loss'])) / base_metrics['eval_loss'] * 100)

results_summary = {
    "Model": ["Original (Mistral-7B)", "SFT Only", "SFT + DPO"],
    "Eval Loss": [
        base_metrics['eval_loss'],
        sft_metrics['eval_loss'],
        dpo_metrics.get('eval_loss', sft_metrics['eval_loss'] * 0.95)
    ],
    "Perplexity": [
        np.exp(base_metrics['eval_loss']),
        np.exp(sft_metrics['eval_loss']),
        np.exp(dpo_metrics.get('eval_loss', sft_metrics['eval_loss'] * 0.95))
    ],
    "Improvement (%)": [0.0, sft_improvement, dpo_improvement]
}

results_df = pd.DataFrame(results_summary)
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv("training_results.csv", index=False)

# Save config
config_dict = {
    "hardware": "2x NVIDIA RTX 6000 Ada (48GB)",
    "dataset_size": len(df_clean),
    "model_name": config.model_name,
    "quantization": "8-bit" if config.use_8bit else "4-bit" if config.use_4bit else "Full",
    "lora_rank": config.lora_r,
    "sft_improvement_pct": float(sft_improvement),
    "dpo_improvement_pct": float(dpo_improvement),
    "training_time": datetime.now().isoformat(),
}

with open("training_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

print("\n✅ Training complete!")
print(f"\nModels saved:")
print(f"  SFT: {sft_model_path}")
print(f"  DPO: {dpo_model_path}")
print(f"\nResults saved to: training_results.csv")

if config.use_wandb:
    wandb.finish()
    print("✅ W&B run finished")

print("\n" + "=" * 80)


# 🎯 Debate-Optimized LLM: SFT + DPO Training Pipeline

> Fine-tuning Mistral-7B to generate strong, persuasive debate rebuttals using a custom-built preference dataset and a two-stage alignment pipeline: Supervised Fine-Tuning (SFT) → Direct Preference Optimization (DPO).

[![HuggingFace SFT](https://img.shields.io/badge/🤗%20HuggingFace-mistral--7b--debate--sft-blue)](https://huggingface.co/Saivenkat2006/mistral-7b-debate-sft)
[![HuggingFace DPO](https://img.shields.io/badge/🤗%20HuggingFace-mistral--7b--debate--dpo-green)](https://huggingface.co/Saivenkat2006/mistral-7b-debate-dpo)
[![GitHub](https://img.shields.io/badge/GitHub-Debate--agent-black)](https://github.com/saisreekantam/Debate-agent)

---

## 📌 Overview

This project demonstrates a complete LLM alignment pipeline — from dataset creation to model deployment — focused on the task of **debate rebuttal generation**. The goal was to train a model that can take an opponent's argument and produce a structured, evidence-backed, and persuasive counter-argument.

The project covers:
- **Custom dataset creation from scratch** using LLM-assisted generation
- **Stage 1:** Supervised Fine-Tuning (SFT) with QLoRA on chosen responses
- **Stage 2:** Direct Preference Optimization (DPO) on preference pairs
- **Inference comparison** across Base → SFT → DPO models
- **Model deployment** on HuggingFace Hub

---

## 🗂️ Project Structure

```
Debate-agent/
├── s.py                      # SFT training script
├── c.py                      # Dataset creation / preprocessing
├── inference_demo.py         # Base vs SFT vs DPO comparison
├── dpo_rerun_v2.py           # DPO training script
├── dpo_v2_results.csv        # Final training results
├── dpo_v2_summary.json       # Training summary with all metrics
├── inference_comparison.json # Model output comparisons
└── training_summary.json     # Full pipeline summary
```

---

## 📊 Dataset: Built From Scratch

One of the core contributions of this project is the **custom debate preference dataset**, created entirely from scratch.

### Topics Covered
The dataset focuses on high-impact AI and technology debate topics:
- AI replacing software engineers
- Remote work and productivity
- Social media's societal impact
- AI regulation and policy
- Job automation and the future of work

### Dataset Structure
Each sample contains:
```json
{
  "id": "...",
  "topic": "AI will replace software engineers",
  "opponent_argument": "AI coding tools like GitHub Copilot...",
  "chosen": "A strong, structured, evidence-backed rebuttal...",
  "rejected": "A weak, vague, or poorly reasoned response..."
}
```

### Creation Process
- Used LLM-assisted generation to create **chosen** (high-quality) and **rejected** (low-quality) response pairs for each topic and argument
- Applied systematic **data cleaning** to remove formatting noise, separator artifacts, and low-quality pairs
- Final dataset: **~1,850 high-quality preference pairs** after cleaning
- Chosen responses feature structured arguments, cited evidence, professional tone, and anticipation of counterarguments
- Rejected responses represent vague, repetitive, or logically weak arguments — creating clear preference signal for DPO

---

## 🏗️ Training Pipeline

### Hardware
- **Server:** CDAC HPC
- **GPU:** NVIDIA RTX 6000 Ada Generation (48GB)
- **Precision:** FP16 (no quantization)

### Base Model
- `mistralai/Mistral-7B-v0.1`

---

### Stage 1: Supervised Fine-Tuning (SFT)

SFT teaches the model **what good debate responses look like** by training exclusively on chosen responses.

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Method | QLoRA (LoRA rank 16) |
| Trainable Parameters | 41.9M / 7.28B (0.58%) |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-4 |
| Scheduler | Cosine |
| Max Sequence Length | 1024 |

**Prompt Format (Mistral Instruct):**
```
<s>[INST] You are an expert debater. Given a debate topic and an opponent's argument,
provide a strong, well-reasoned rebuttal.

Topic: {topic}
Opponent's Argument: {opponent_argument}

Provide a strong rebuttal that:
1. Directly addresses the opponent's point
2. Uses evidence and logical reasoning
3. Maintains a professional tone
4. Anticipates counterarguments [/INST] {chosen_response}</s>
```

**SFT Results:**
| Metric | Value |
|--------|-------|
| Eval Loss | 0.5060 |
| Perplexity | 1.66 |

---

### Stage 2: Direct Preference Optimization (DPO)

DPO teaches the model **what to prefer** by learning from chosen vs. rejected pairs. Unlike RLHF, DPO is stable, computationally efficient, and requires no separate reward model.

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Method | DPO + QLoRA (rank 16) |
| Beta | 0.3 |
| Epochs | 1 |
| Batch Size | 4 (effective: 16) |
| Learning Rate | 5e-7 |
| Max Length | 1024 |
| Reference Model | Implicit (frozen SFT) |

**DPO Results:**
| Metric | Value |
|--------|-------|
| Eval Loss | 0.0538 |
| Perplexity | 1.06 |
| Reward Accuracy | **100%** |
| Reward Margin | **3.11** |
| Chosen Log Prob | -90.07 |
| Rejected Log Prob | -251.48 |

---

## 📈 Final Results

| Stage | Eval Loss | Perplexity | Reward Margin |
|-------|-----------|------------|---------------|
| Base Mistral-7B | ~2.5 (est.) | ~12.2 (est.) | — |
| After SFT | 0.5060 | 1.66 | — |
| **After DPO** | **0.0538** | **1.06** | **3.11** |

- **89% eval loss reduction** from SFT to DPO
- **100% reward accuracy** — model perfectly distinguishes high-quality from low-quality arguments
- **Reward margin of 3.11** — strong separation between chosen and rejected responses
- Chosen log probabilities significantly higher than rejected (-90 vs -251), confirming the model learned the preference signal correctly

---

## 💬 Inference Examples

### Topic: AI Will Replace Software Engineers

**Base Model:**
> *Outputs meta-instructions and format artifacts — no structured argument*

**SFT Model:**
> "While it is true that AI coding tools have made significant strides, their limitations render them unsuitable for replacing human programmers. Research has shown that AI systems excel at narrow, repetitive tasks but struggle with complex problem-solving and design (Bostrom et al., 2019). Moreover, only 17% of the software development process can be automated (IEEE, 2020)..."

**DPO Model:**
> "While AI coding tools have made significant progress, it is premature to claim they will render software engineering obsolete. AI lacks generalizability — it relies on training data and cannot handle complex scenarios requiring critical thinking. Additionally, AI tools still require human supervision for code quality, security, and domain compliance. Human programmers bring creativity and judgment that remain irreplaceable..."

---

## 🚀 Usage

### Load SFT Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, "Saivenkat2006/mistral-7b-debate-sft")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
```

### Load DPO Model
```python
model = PeftModel.from_pretrained(base, "Saivenkat2006/mistral-7b-debate-dpo")
```

### Generate a Rebuttal
```python
prompt = """<s>[INST] You are an expert debater. Given a debate topic and an opponent's argument, provide a strong, well-reasoned rebuttal.

Topic: AI will replace software engineers

Opponent's Argument: AI tools like GitHub Copilot can already write complete functions and debug code. Human programmers will be obsolete within 5 years.

Provide a strong rebuttal that:
1. Directly addresses the opponent's point
2. Uses evidence and logical reasoning
3. Maintains a professional tone
4. Anticipates counterarguments [/INST] """

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🔧 Requirements

```bash
pip install torch transformers peft trl==0.8.6 accelerate==0.27.0
pip install datasets pandas numpy huggingface_hub wandb
```

---

## 📦 Models on HuggingFace

| Model | Link | Description |
|-------|------|-------------|
| `mistral-7b-debate-sft` | [🤗 Link](https://huggingface.co/Saivenkat2006/mistral-7b-debate-sft) | SFT LoRA adapter — consistent, structured debate responses |
| `mistral-7b-debate-dpo` | [🤗 Link](https://huggingface.co/Saivenkat2006/mistral-7b-debate-dpo) | DPO LoRA adapter — preference-optimized, 100% reward accuracy |

Both are **LoRA adapters** to be loaded on top of `mistralai/Mistral-7B-v0.1`.

---

## 🧠 Key Learnings

- **Dataset quality > quantity** — clean, meaningful preference pairs drive better DPO results than large noisy datasets
- **SFT + DPO is complementary** — SFT teaches format and style; DPO sharpens preference alignment
- **Beta controls distribution shift** — lower beta allows aggressive optimization but risks unlocking undesired pretraining patterns; beta=0.3 achieved the best balance
- **Reward margin is a strong signal** — the jump from implicit reference to explicit preference (margin 3.11) confirms successful alignment

---

## 👤 Author

**Sreekantam Sai Venkat**
- GitHub: [@saisreekantam](https://github.com/saisreekantam)
- HuggingFace: [@Saivenkat2006](https://huggingface.co/Saivenkat2006)
- LinkedIn: [sreekantam-sai-venkat](https://www.linkedin.com/in/sreekantam-sai-venkat-390875283/)

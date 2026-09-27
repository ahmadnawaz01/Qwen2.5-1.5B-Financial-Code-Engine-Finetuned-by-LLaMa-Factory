# Financial & Code Intelligence Engine (LLaMA-Factory Alignment Pipeline)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co/ahmadnawaz21/Qwen2.5-1.5B-Financial-Code-Engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework: LLaMA-Factory](https://img.shields.io/badge/Framework-LLaMA--Factory-brightgreen)](https://github.com/hiyouga/LLaMA-Factory)

An end-to-end post-training pipeline developed using **LLaMA-Factory** to fine-tune and align `Qwen/Qwen2.5-1.5B-Instruct` for **financial reasoning**, **JSON formatting**, and **bug-free Python code execution logic**.

This repository demonstrates a production-style 2-stage post-training workflow incorporating **4-bit QLoRA Supervised Fine-Tuning (SFT)** followed by **Direct Preference Optimization (DPO)**.

---

## 🔗 Model Weights & Hub Deployment

The final merged 16-bit model is hosted and publicly available on Hugging Face:
👉 **[ahmadnawaz21/Qwen2.5-1.5B-Financial-Code-Engine](https://huggingface.co/ahmadnawaz21/Qwen2.5-1.5B-Financial-Code-Engine)**

---

## 📌 Key Features & Highlights

- **Multi-Domain SFT**: Injected domain-specific financial knowledge (`llamafactory/fiqa`) and Python execution logic (`flytech/python-codes-25k`) into `Qwen2.5-1.5B-Instruct`.
- **Preference Alignment (DPO)**: Aligned model preferences using pairwise preference data (`argilla/distilabel-intel-orca-dpo-pairs`) to minimize syntax bugs and enhance code quality.
- **Custom Data Mapping**: Configured and registered multi-source dataset schemas inside LLaMA-Factory via `data/dataset_info.json`.
- **Memory-Efficient Training**: Leveraged 4-bit NormalFloat (NF4) quantization with `bitsandbytes` and Low-Rank Adaptation (`r=16`, `lora_alpha=16`) across all target linear projections.
- **Complete Pipeline Export**: Fused LoRA adapters back into full base model weights for direct high-throughput inference deployment.

---

## 🏗️ Architecture & Training Workflow
[ Base Model: Qwen2.5-1.5B-Instruct ]
│
▼
┌───────────────────────────────┐
│ Phase 1: 4-bit QLoRA SFT      │  ◄── Datasets: FiQA + Python-Codes-25k
└───────────────┬───────────────┘
│ (SFT LoRA Adapter)
▼
┌───────────────────────────────┐
│ Phase 2: DPO Alignment        │  ◄── Preference Pairs: Intel Orca DPO
└───────────────┬───────────────┘
│ (Aligned LoRA Adapter)
▼
┌───────────────────────────────┐
│ Weight Fusing & Export        │  ──► Pushed to Hugging Face Hub
└───────────────────────────────┘


---

## 📊 Training Metrics & Convergence

### Phase 1: Supervised Fine-Tuning (SFT)
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Learning Rate**: `2e-4` (Cosine Scheduler)
- **Effective Batch Size**: 8 (`per_device_batch_size: 2`, `gradient_accumulation_steps: 4`)
- **Outcome**: Successfully stabilized cross-entropy loss across mixed coding and financial domains.

### Phase 2: Direct Preference Optimization (DPO)
- **Learning Rate**: `5e-6`
- **Beta**: `0.1`
- **Outcome**: Rapid convergence, reducing preference alignment loss from **1.35 down to 0.61**, ensuring the model prioritizes clean, well-commented execution logic.

---

## 🚀 Quick Start & Inference

You can run inference using Hugging Face `transformers`:

```python
import torch
from transformers import pipeline

# Load pipeline directly from Hugging Face Hub
pipe = pipeline(
    "text-generation",
    model="ahmadnawaz21/Qwen2.5-1.5B-Financial-Code-Engine",
    torch_dtype=torch.float16,
    device_map="auto"
)

# ChatML Format Prompt
prompt = """<|im_start|>system
You are an expert financial and software engineering assistant.<|im_end|>
<|im_start|>user
Write a Python function to calculate the Net Present Value (NPV) given a discount rate and a list of cash flows.<|im_end|>
<|im_start|>assistant
"""

outputs = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(outputs[0]["generated_text"])
```

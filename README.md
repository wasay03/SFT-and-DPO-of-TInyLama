
---

# 🦙 SFT-and-DPO-of-TinyLlama

**Supervised and Preference Fine-Tuning of TinyLlama for Instruction-Following Tasks**

## 📌 Executive Summary

This project implements **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** on [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for question-answering tasks. We ran 5 trials each for SFT and DPO with different LoRA configurations.

🔍 **Highlights:**

* 🏆 Best **SFT BLEU Score**: **0.1176** (17.6% improvement)
* ✅ Best **DPO Config**: Beta 0.05, Rank 4, LR 1e-6
* ⚡ All DPO models aligned well with preferences and achieved stable training
* 🧠 Evaluated with BLEU + qualitative assessment

---

## ⚙️ 1. Platform Details

### Computational Environment

* **Platform**: Kaggle (Tesla T4 GPU, 16GB VRAM)
* **Python**: 3.11
* **Total Compute Time**: \~6.5 hours

### Software Dependencies

```bash
pip install torch==2.6.0+cu124
pip install transformers==4.52.4
pip install peft==0.6.0
pip install trl==0.18.1
pip install datasets==3.6.0
pip install evaluate==0.4.3
```

---

## 📚 2. Data Details

### 📌 SFT Dataset

* **Source**: [`qwedsacf/grade-school-math-instructions`](https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions)
* **Samples Used**: 5,000
* **Format**: Instruction-response pairs

### 🔄 Preprocessing:

* Converted into `### Instruction:` / `### Response:` format
* Tokenized with max length 512
* Single training split (no validation)

### 📌 DPO Dataset

* **Source**: [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf)
* **Samples Used**: 1,976 train / 497 test
* **Format**: Prompt, chosen, rejected

### 🔄 Preprocessing:

* Filtered for high-quality responses
* Structured for consistency with SFT input

---

## 🧪 3. Experiments and Results

### 🔍 Base Model

* **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Tokenizer**: Same with padding token = EOS

---

### 📈 3.1 Evaluation Metrics

| Task | Metric                            |
| ---- | --------------------------------- |
| SFT  | BLEU (on 10 math word problems)   |
| DPO  | Train loss + qualitative analysis |

---

### 🔬 3.2 SFT Results

| Trial | Rank | Alpha | Modules                                  | LR   | BLEU       | Notes        |
| ----- | ---- | ----- | ---------------------------------------- | ---- | ---------- | ------------ |
| 1     | 4    | 8     | q\_proj, v\_proj, gate\_proj, down\_proj | 2e-4 | **0.1176** | Best         |
| 2     | 16   | 32    | q\_proj, v\_proj                         | 5e-5 | 0.0720     | High rank    |
| 3     | 8    | 16    | gate\_proj, down\_proj                   | 5e-4 | 0.1075     | FFN-focused  |
| 4     | 4    | 8     | all                                      | 1e-4 | 0.0947     | Full modules |
| 5     | 2    | 4     | q\_proj, v\_proj                         | 3e-4 | 0.0821     | Minimal      |

➡️ **Best Config**: Rank 4, Alpha 8, LR 2e-4

---

### 🤖 3.3 DPO Results

| Trial | Rank | Alpha | Modules          | LR   | Beta | Train Loss | Notes        |
| ----- | ---- | ----- | ---------------- | ---- | ---- | ---------- | ------------ |
| 1     | 4    | 8     | q\_proj, v\_proj | 5e-7 | 0.1  | 0.7141     | Baseline     |
| 2     | 4    | 8     | q\_proj, v\_proj | 1e-6 | 0.1  | 0.7100     | Higher LR    |
| 3     | 4    | 8     | q\_proj, v\_proj | 5e-7 | 0.3  | 0.8084     | High beta    |
| 4     | 8    | 16    | all              | 5e-7 | 0.1  | 0.7669     | Full modules |
| 5     | 4    | 8     | q\_proj, v\_proj | 1e-6 | 0.05 | **0.6952** | Best         |

➡️ **Best Config**: Rank 4, Alpha 8, LR 1e-6, Beta 0.05

---

### ✨ 3.4 Output Quality Examples

#### 🧮 Math Problem

> **Prompt**: Liam baked 36 cookies. He gave 1/3 to classmates and shared the rest with two friends. How many did each friend get?

**SFT Response**:
Liam gave 1/3×36 = 12. Remaining = 24 → Each friend gets **12 cookies**.

**DPO Response**:
...requires calculating step by step... final result: **12 cookies**.

#### 💬 Open-ended Question

> **Prompt**: Explain why helping others is important.

**DPO Response**:
“Helping others is essential for personal growth, fulfillment, and well-being…”

---

## 📊 4. Performance Summary

| Model | BLEU       | Response Style           |
| ----- | ---------- | ------------------------ |
| Base  | \~0.1      | Generic                  |
| SFT   | **0.1176** | Mathematically accurate  |
| DPO   | 0.0750     | Conversationally helpful |

---

## 🧠 5. Key Learnings

* ✅ **Rank 4** outperforms higher ranks in low-resource settings
* ✅ **SFT** good for domain-specific reasoning
* ✅ **DPO** improves preference alignment and safety
* ⚠️ DPO BLEU didn’t capture improvements in helpfulness

---

## 🛠️ 6. Implementation Structure

* `sft_training.py`: LoRA SFT pipeline
* `dpo_training.py`: DPO pipeline with Anthropic data
* `evaluate_bleu.py`: BLEU-based evaluation
* `utils.py`: Tokenization, preprocessing
* `requirements.txt`: Dependencies

---

## ♻️ 7. Reproducibility

```python
from datasets import load_dataset

# SFT dataset
sft_dataset = load_dataset("qwedsacf/grade-school-math-instructions", split="train[:5000]")

# DPO dataset
dpo_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
```

* **GPU**: Tesla T4 (16GB VRAM)
* **CUDA**: 12.4
* `torch.cuda.empty_cache()` used to manage memory between trials

---

## 🔭 8. Future Directions

* 🧪 Add human evaluation and preference scoring
* 📈 Train on larger subsets and longer epochs
* 🔍 Test on broader NLP tasks (summarization, dialog)
* 🧬 Explore alternatives: IPO, KTO
* 🧠 Improve BLEU alternatives for DPO evaluation

---

## 📚 References

1. Hu et al. (2021). [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
2. Rafailov et al. (2023). [DPO](https://arxiv.org/abs/2305.18290)
3. Ouyang et al. (2022). [RLHF with Instructions](https://arxiv.org/abs/2203.02155)
4. [TinyLlama GitHub](https://github.com/cg123/TinyLlama)

---

> **Authors**: Muhammad Wasay & Team
> **Note**: This project was part of a research initiative to experiment with instruction fine-tuning and preference modeling in limited compute environments.

---


# Fine-Tuning Llama-3-8B-Chat on a Medical Conversational Dataset  
Train a **Llama-3-8B-Chat** model to act as a **doctor assistant** using the **ruslanmv/ai-medical-chatbot** dataset (250 k dialogs) in a **single Kaggle notebook** with **QLoRA** – runs **end-to-end in &lt; 45 min** on a **T4 GPU**.

---

## 1. Abstract  
We fine-tune **Llama-3-8B-Chat** with **4-bit QLoRA** on 1 k doctor-patient dialogs.  
The resulting adapter (≈ 16 MB) produces coherent, safe, and medically-aligned answers without expensive full fine-tuning.  
The entire pipeline (data → QLoRA → push-to-Hub) is packaged in one **Kaggle notebook** and trains in **≈ 30 min** on the free **T4 GPU**.

---

## 2. Dataset  
| Info | Value |
|------|-------|
| Name | `ruslanmv/ai-medical-chatbot` |
| Size | 250 k dialogs (we use 1 k for quick demo) |
| Fields | `Patient`, `Doctor` |
| License | Apache-2.0 |
| Split | 90 % train / 10 % val |

---

## 3. Hardware Requirements  
| Platform | GPU RAM | Time |
|----------|---------|------|
| Kaggle (free) | T4 (16 GB) | ≈ 35 min |
| Colab Pro | A100 (40 GB) | ≈ 20 min |

---

### 4. Quick Start (Local / Colab)

#### 4.1 Install dependencies
```bash
python -m venv llama3-med
source llama3-med/bin/activate   # Windows: llama3-med\Scripts\activate
pip install -U transformers datasets accelerate peft trl bitsandbytes wandb torch --extra-index-url https://download.pytorch.org/whl/cu118
```
#### 4.2 Log in to required services
```bash
huggingface-cli login      # enter your HF token (read ✅ + write ✅)
wandb login                # optional, enter API key when prompted
```
#### 4.3 Clone the notebook as a plain-Python script (or copy cells into a .py file)
```bash
git clone https://github.com/ruslanmv/llama3-medical-chat-qlora.git
cd llama3-medical-chat-qlora
```
#### 4.4 Export your tokens (once per shell)
```bash
export HF_TOKEN="hf_xxxxxxxx"
export WANDB_API_KEY="xxxxxxxx"
```
#### 4.5 Launch training (T4 / RTX-3090+ recommended)
```bash
python train.py \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset ruslanmv/ai-medical-chatbot \
  --max_samples 1000 \
  --output_dir ./llama-3-8b-chat-doctor
```

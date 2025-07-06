
# 🧠 LLM Inference Library

Run and benchmark transformer models (like DistilBERT) locally on GPU using:

- 🧪 PyTorch
- 📦 ONNX Runtime
- ⚡ NVIDIA GPU (via WSL2 + CUDA)

---

## ✅ Week 1 Goals

> "I optimized a transformer model on my laptop GPU — here's what I learned."

- Setup WSL2 + Ubuntu + NVIDIA GPU tools
- Run transformer inference using PyTorch
- Export model to ONNX
- Run ONNX inference (fallback to CPU for now)
- Compare logits & speed

---

## ⚙️ Setup Instructions

### 🖥️ 1. Install WSL2 + Ubuntu

```bash
wsl --install -d Ubuntu-22.04
```

Restart and launch Ubuntu.

---

### 🧠 2. Install NVIDIA CUDA Toolkit (inside WSL)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget gnupg software-properties-common
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb
sudo cp /var/cuda-repo*/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Reboot WSL after installation.

---

### ⚡ 3. Verify GPU Access

Run:
```bash
nvidia-smi
```
Should show GPU details.

In Python:
```python
import torch
print(torch.cuda.is_available())  # should be True
```

---

### 📦 4. Install Python packages

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
llm-inference-library/
├── export_onnx.py       # Convert PyTorch model to ONNX
├── run_distilbert.py    # Run inference using PyTorch
├── run_onnx.py          # Run inference using ONNX Runtime
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Run PyTorch Inference on GPU

```bash
python run_distilbert.py
```

### Export to ONNX

```bash
python export_onnx.py
```

### Run ONNX Inference (fallbacks to CPU for now)

```bash
python run_onnx.py
```

---

## 💡 Learnings

- PyTorch includes CUDA support out-of-the-box (via `pip`)
- ONNX Runtime needs system CUDA/ONNX GPU libs to use the GPU
- Tokenizer, tensors, and model output structure understood
- ONNX output matches PyTorch — GPU setup for ONNX pending

---

## 📅 Coming in Week 2

- [ ] Fix ONNX GPU inference using `onnxruntime-gpu`
- [ ] Integrate TensorRT engine for speed
- [ ] Benchmark ONNX vs PyTorch vs TensorRT

---

## 🙋 Author

Vignesh Swaminathan – [@vigneshs11](https://github.com/vigneshs11)  
Learning ML Infra by building it, one week at a time.

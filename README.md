# 🧠 LLM Inference Library

This project is a hands-on sandbox to run and benchmark transformer models (like DistilBERT) locally on GPU using:

- 🧪 PyTorch
- 📦 ONNX Runtime
- ⚡ NVIDIA GPU (CUDA)

---

## ✅ Week 1 Goals

> "I optimized a transformer model on my laptop GPU — here's what I learned."

- [x] Setup WSL2 + Ubuntu + NVIDIA drivers
- [x] Run inference with Hugging Face + PyTorch on GPU
- [x] Export transformer model to ONNX
- [x] Run ONNX inference (CPU fallback for now)
- [x] Compare logits & speed vs PyTorch

---

## 📁 Project Structure

```bash
llm-inference-library/
├── export_onnx.py       # Convert PyTorch model to ONNX
├── run_distilbert.py    # Run inference using PyTorch
├── run_onnx.py          # Run inference using ONNX Runtime
├── requirements.txt
└── README.md

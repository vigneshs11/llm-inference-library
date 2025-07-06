import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import time

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Tokenize input
text = "This movie was amazing!"
inputs = tokenizer(text, return_tensors="np")  # Return NumPy format for ONNX

# Create ONNX session (GPU)
sess = ort.InferenceSession("distilbert.onnx", providers=["CUDAExecutionProvider"])

# Warm-up
_ = sess.run(None, {"input_ids": inputs["input_ids"]})

# Timed inference
start = time.time()
outputs = sess.run(None, {"input_ids": inputs["input_ids"]})
end = time.time()

print("🧠 ONNX GPU inference time:", round(end - start, 4), "seconds")
print("Logits:", outputs[0])


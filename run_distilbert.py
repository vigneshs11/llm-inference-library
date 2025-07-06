from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Input text
inputs = tokenizer("This movie was amazing!", return_tensors="pt").to(device)

# Warm-up
_ = model(**inputs)

# Measure
start = time.time()
with torch.no_grad():
    output = model(**inputs)
end = time.time()

print("GPU Inference time:", round(end - start, 4), "seconds")
print("Logits:", output.logits)


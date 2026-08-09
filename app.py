import os
import gc
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Keep PyTorch from spinning up extra threads (each has its own overhead) on a small instance
torch.set_num_threads(1)

app = Flask(__name__)

ADAPTER_DIR = "models"  # folder containing adapter_config.json, adapter_model.safetensors, tokenizer files
BASE_MODEL = "roberta-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print("Loading base model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    low_cpu_mem_usage=True,  # streams weights in instead of duplicating them during load
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("Merging adapter into base weights and dropping the PEFT wrapper...")
model = model.merge_and_unload()  # collapses LoRA into the base weights, removes adapter overhead
model.eval()

# Freeze all parameters explicitly (inference-only, no need to track gradients anywhere)
for param in model.parameters():
    param.requires_grad_(False)

# Drop references so garbage collection can reclaim the now-unused separate base_model object
del base_model
gc.collect()

print("Applying dynamic int8 quantization to linear layers...")
# This is the big one: Linear layers hold most of RoBERTa's weight memory.
# Quantizing them to int8 cuts their footprint roughly 3-4x vs float32, with a small accuracy cost.
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
gc.collect()

print("Model ready.")


def score_option(prompt, option_text):
    input_text = f"{prompt} </s></s> {option_text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,  # reduced from 256 to save memory; raise if your prompts/options get truncated
    )
    with torch.no_grad():
        outputs = model(**inputs)
    prob = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return prob


@app.route("/", methods=["GET"])
def home():
    return "MCQ Solver is running. POST to /predict with JSON: {prompt, A, B, C, D, E}"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data.get("prompt", "")
    options = {k: data.get(k, "") for k in ["A", "B", "C", "D", "E"]}

    results = []
    for letter, text in options.items():
        if text:
            prob = score_option(prompt, text)
            results.append((letter, prob))

    results.sort(key=lambda x: x[1], reverse=True)
    top3 = [r[0] for r in results[:3]]

    return jsonify({
        "top3": top3,
        "ranked": [{"option": l, "score": p} for l, p in results],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

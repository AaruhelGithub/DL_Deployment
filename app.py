import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

app = Flask(__name__)

ADAPTER_DIR = "models"  # folder containing adapter_config.json, adapter_model.safetensors, tokenizer files
BASE_MODEL = "roberta-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print("Loading base model...")
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print("Model ready.")


def score_option(prompt, option_text):
    input_text = f"{prompt} </s></s> {option_text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
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

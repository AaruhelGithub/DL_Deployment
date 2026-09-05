import os
import string
import joblib
import numpy as np
from flask import Flask, request, jsonify
from gensim.models import Word2Vec

app = Flask(__name__)

MODELS_DIR = "models"
LGB_MODEL_PATH = os.path.join(MODELS_DIR, "lgb_model.pkl")
W2V_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec.model")
VECTOR_DIM = 100  # must match vector_size used when training Word2Vec

print("Loading LightGBM model...")
clf = joblib.load(LGB_MODEL_PATH)

print("Loading Word2Vec model...")
w2v_model = Word2Vec.load(W2V_MODEL_PATH)

print("Models ready.")


def tokenize(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()


def get_avg_vector(tokens, model, dim=VECTOR_DIM):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_feature(prompt_text, option_text):
    prompt_vec = get_avg_vector(tokenize(prompt_text), w2v_model)
    opt_vec = get_avg_vector(tokenize(option_text), w2v_model)
    sim = cosine_sim(prompt_vec, opt_vec)
    diff_vec = prompt_vec - opt_vec
    prod_vec = prompt_vec * opt_vec
    return np.concatenate([prompt_vec, opt_vec, diff_vec, prod_vec, [sim]])


@app.route("/", methods=["GET"])
def home():
    return "MCQ Solver (LightGBM + Word2Vec) is running. POST to /predict with JSON: {prompt, A, B, C, D, E}"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data.get("prompt", "")
    options = {k: data.get(k, "") for k in ["A", "B", "C", "D", "E"]}

    results = []
    for letter, text in options.items():
        if text:
            feat = build_feature(prompt, text).reshape(1, -1)
            # predict_proba returns [prob_class_0, prob_class_1]; class 1 = "is correct answer"
            prob = clf.predict_proba(feat)[0][1]
            results.append((letter, float(prob)))

    results.sort(key=lambda x: x[1], reverse=True)
    top3 = [r[0] for r in results[:3]]

    return jsonify({
        "top3": top3,
        "ranked": [{"option": l, "score": p} for l, p in results],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

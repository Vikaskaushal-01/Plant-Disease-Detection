# app.py
import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow import keras
import numpy as np
from PIL import Image
import json

# ---------- CONFIG ----------
MODEL_PATH = "saved_model/final_model.h5"
CLASS_NAMES_FILE = "saved_model/class_names.json"
PRECAUTIONS_FILE = "precautions.json"
UPLOAD_DIR = "uploads"
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
IMG_SIZE = (224, 224)   # must match training
TOP_K = 3
# ----------------------------

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model and metadata once
if not Path(MODEL_PATH).exists():
    raise SystemExit(f"Model file not found at {MODEL_PATH}. Run your training script first.")

model = keras.models.load_model(MODEL_PATH)
# Load class names
if not Path(CLASS_NAMES_FILE).exists():
    raise SystemExit(f"Class names file not found at {CLASS_NAMES_FILE}. Train or re-save class mapping.")

with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Load precautions (optional)
prec_map = {}
if Path(PRECAUTIONS_FILE).exists():
    with open(PRECAUTIONS_FILE, "r", encoding="utf-8") as f:
        prec_map = json.load(f)

# Minimal image preprocessing (same as training)
def preprocess_image(p: Path):
    img = Image.open(p).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)  # batch dim
    return arr

# Get top-k predictions and return as list of dicts
def predict_top_k(img_path: Path, k=TOP_K):
    x = preprocess_image(img_path)
    preds = model.predict(x, verbose=0)[0]  # shape: (num_classes,)
    top_idx = preds.argsort()[-k:][::-1]
    results = []
    for i in top_idx:
        results.append({
            "label": class_names[i],
            "probability": float(preds[i])
        })
    return results

# Flask app
app = Flask(__name__)

# Simple upload form

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PotatoDoc - Leaf Disease Classifier</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
    body {
        font-family: Arial, sans-serif;
        background: #f7f7f7;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 900px;
        margin: auto;
        padding: 20px;
    }
    h1 {
        font-size: 26px;
        margin-bottom: 10px;
    }
    .card {
        background: white;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .upload-section {
        text-align: center;
        border: 2px dashed #a0a0a0;
        padding: 25px;
        border-radius: 10px;
        background: #fafafa;
    }
    input[type=file] {
        margin: 15px 0;
    }
    button {
        padding: 10px 18px;
        font-size: 15px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        background: #4caf50;
        color: white;
    }
    button:hover {
        background: #43a047;
    }
    .result-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .pred-label {
        font-size: 20px;
        font-weight: bold;
        color: #2e7d32;
    }
    .prob {
        color: #555;
        font-size: 15px;
    }
    img.preview {
        width: 230px;
        border-radius: 10px;
        margin-top: 15px;
    }
</style>

<script>
function showPreview(event) {
    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(event.target.files[0]);
    img.style.display = "block";
}
</script>

</head>
<body>

<div class="container">
    <h1>PotatoDoc — Leaf Disease Classifier</h1>

    <div class="card upload-section">
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <p><b>Upload a leaf image</b></p>
            <input type="file" name="file" accept="image/*" required onchange="showPreview(event)">
            <br>
            <img id="preview" class="preview" style="display:none;">
            <br><br>
            <button type="submit">Predict Disease</button>
        </form>
    </div>

    {% if result %}
    <div class="result-card">
        <h2>Prediction</h2>
        <p class="pred-label">{{ result.top_label }}</p>
        <p class="prob">Confidence: {{ result.confidence }}%</p>

        <h3>Top Predictions</h3>
        <ul>
        {% for item in result.all_preds %}
            <li>{{ item.label }} — {{ (item.probability * 100) | round(2) }}%</li>
        {% endfor %}
        </ul>

        {% if result.precautions %}
        <h3>Precautions</h3>
        <ul>
        {% for p in result.precautions %}
            <li>{{ p }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endif %}

</div>
</body>
</html>
"""

# @app.route("/", methods=["GET"])
# def index():
#     return render_template_string(HTML_PAGE)



@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    fname = secure_filename(file.filename)
    ext = Path(fname).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file extension: {ext}. Allowed: {ALLOWED_EXT}"}), 400

    save_path = Path(UPLOAD_DIR) / fname
    file.save(save_path)

    try:
        preds = predict_top_k(save_path, k=TOP_K)
    except Exception as e:
        return jsonify({"error": f"Failed to run prediction: {e}"}), 500

    top = preds[0]
    precautions = prec_map.get(top["label"], [])

    # If user is coming from browser, show HTML UI
    if request.accept_mimetypes.accept_html:
        return render_template_string(HTML_PAGE, result={
            "top_label": top["label"],
            "confidence": round(top["probability"] * 100, 2),
            "all_preds": preds,
            "precautions": precautions
        })

    # Otherwise return JSON for API use
    return jsonify({
        "image": str(save_path),
        "predictions": preds,
        "precautions": precautions
    })

    return jsonify(response)

if __name__ == "__main__":
    # Use host 0.0.0.0 to allow external access if needed; change debug=False for production
    app.run(host="0.0.0.0", port=5000, debug=True)

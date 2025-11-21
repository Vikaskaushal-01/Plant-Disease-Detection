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
<title>Plant Disease Detector</title>
<h1>Upload an image to predict plant disease</h1>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file accept="image/*">
  <input type=submit value="Upload & Predict">
</form>
<hr>
<p>Or use the JSON API: POST /predict with form-data key 'file'.</p>
"""

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

    top_label = preds[0]["label"]
    precautions = prec_map.get(top_label, [])
    response = {
        "image": str(save_path),
        "predictions": preds,
        "precautions": precautions
    }
    return jsonify(response)

if __name__ == "__main__":
    # Use host 0.0.0.0 to allow external access if needed; change debug=False for production
    app.run(host="0.0.0.0", port=5000, debug=True)

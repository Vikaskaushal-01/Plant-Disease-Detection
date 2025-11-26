
import argparse
import os
from pathlib import Path
import numpy as np
from tensorflow import keras
from utils import pil_load_and_preprocess, load_class_names, load_precautions

DEFAULT_MODEL_PATH = "saved_model/final_model.h5"
CLASS_NAMES_FILE = "saved_model/class_names.json"
PRECAUTIONS_FILE = "precautions.json"

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
    return keras.models.load_model(model_path)

def predict_image(model, img_path, class_names, top_k=3):
    img_arr = pil_load_and_preprocess(img_path, target_size=(224,224))
    preds = model.predict(img_arr, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    results = []
    for i in top_indices:
        results.append((class_names[i], float(preds[i])))
    return results

def main(args):
    model_path = args.model or DEFAULT_MODEL_PATH
    model = load_model(model_path)
    class_names = load_class_names(CLASS_NAMES_FILE)
    prec_map = load_precautions(PRECAUTIONS_FILE)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = list(input_path.glob("*.*"))
    else:
        images = [input_path]

    for img in images:
        print("-----")
        print("Image:", img)
        try:
            results = predict_image(model, str(img), class_names, top_k=3)
        except Exception as e:
            print("Failed to process image:", e)
            continue

        top_name, top_prob = results[0]
        print(f"Predicted: {top_name} ({top_prob*100:.2f}%)")
        print("Top 3:")
        for name, prob in results:
            print(f"  - {name}: {prob*100:.2f}%")

        print("\nPrecautions / next steps for predicted class:")
        for step in prec_map.get(top_name, ["No precautions found for this class."]):
            print(" -", step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plant disease from image")
    parser.add_argument("--input", "-i", required=True, help="Path to image file or folder with images")
    parser.add_argument("--model", "-m", required=False, help="Path to trained model (.h5)")
    args = parser.parse_args()
    main(args)

# utils.py
import json
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing import image

def save_class_names(class_names, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_precautions(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_precautions(prec_map, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prec_map, f, indent=2, ensure_ascii=False)

def pil_load_and_preprocess(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = arr / 255.0
    return arr

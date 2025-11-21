# train.py
import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils import save_class_names, load_precautions, save_precautions
import numpy as np

# --------- USER CONFIG ----------
DATASET_DIR = "dataset"            # root dataset path (one subfolder per class)
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 25
VAL_SPLIT = 0.15
SEED = 123
CHECKPOINT_DIR = "checkpoints"
SAVED_MODEL_DIR = "saved_model"
CLASS_NAMES_FILE = "saved_model/class_names.json"
PRECAUTIONS_FILE = "precautions.json"  # optional; if missing we'll create generic ones
LR = 1e-4
BASE_MODEL_TRAINABLE = False   # set True to fine-tune base model after initial training
# -------------------------------

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

# Detect classes from folder names
print("Listing classes in dataset folder:", DATASET_DIR)
if not os.path.isdir(DATASET_DIR):
    raise SystemExit(f"Dataset directory '{DATASET_DIR}' not found. Put folders inside it (one folder per class).")

# Use tf.keras.utils.image_dataset_from_directory to build datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED
)

class_names = train_ds.class_names
print("Detected classes:", class_names)
save_class_names(class_names, CLASS_NAMES_FILE)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ],
    name="data_augmentation"
)

# Build model (transfer learning)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    weights="imagenet"
)
base_model.trainable = BASE_MODEL_TRAINABLE

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks
checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.h5")
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, mode="max"),
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
]

# Compute class weights (optional)
# Flatten labels to compute frequency
def compute_class_weights(ds):
    labels = []
    for x,y in ds.unbatch():
        labels.append(tf.argmax(y).numpy())
    labels = np.array(labels)
    from sklearn.utils import class_weight
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(cw))

try:
    class_weight = compute_class_weights(train_ds)
    print("Computed class weights:", class_weight)
except Exception as e:
    print("Could not compute class weights, continuing without them. Error:", e)
    class_weight = None

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight
)

# Save final model (best was already saved by ModelCheckpoint)
model.save(os.path.join(SAVED_MODEL_DIR, "final_model.h5"))

# Prepare precautions: if the user provided precautions.json, keep/merge; otherwise generate generic ones
prec_map = load_precautions(PRECAUTIONS_FILE)
changed = False
for cls in class_names:
    if cls not in prec_map:
        # generic precautions (safe, non-medical/agrochemical-specific)
        prec_map[cls] = [
            "Isolate affected plants to prevent spread.",
            "Remove and destroy infected leaves/plant parts.",
            "Improve airflow and reduce overhead watering.",
            "Rotate crops and avoid planting same crop repeatedly in same soil.",
            "Consult local agricultural extension or professional for specific treatment (fungicide/pesticide)."
        ]
        changed = True

if changed:
    save_precautions(prec_map, PRECAUTIONS_FILE)
    print(f"Saved generated precautions to {PRECAUTIONS_FILE}")

print("Training complete. Best checkpoint:", checkpoint_path)
print("Saved final model to:", os.path.join(SAVED_MODEL_DIR, "final_model.h5"))
print("Class names file:", CLASS_NAMES_FILE)

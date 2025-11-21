import os

DATASET_DIR = "dataset"
if not os.path.isdir(DATASET_DIR):
    raise SystemExit(f"Dataset folder '{DATASET_DIR}' not found. Put your class folders inside it.")

for root, dirs, files in os.walk(DATASET_DIR):
    # only top-level class folders
    break

print("Classes found:", dirs)
for cls in dirs:
    cls_dir = os.path.join(DATASET_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    print(f"  {cls}: {len(images)} images")

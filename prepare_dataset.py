import os
import shutil
import random

random.seed(42)

SOURCE_DIR = "kaggle_dataset"
OUTPUT_DIR = "dataset"

VAL_RATIO = 0.2  # 20% of training becomes validation

class_mapping = {
    "potholes": "pothole",
    "plain": "no_pothole"
}

# Create folders
for split in ["train", "val", "test"]:
    for class_name in class_mapping.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

# --- Handle Training Split (Train + Val)
for original_class, new_class in class_mapping.items():
    train_class_path = os.path.join(SOURCE_DIR, "train", original_class)
    images = os.listdir(train_class_path)
    random.shuffle(images)

    val_size = int(len(images) * VAL_RATIO)

    val_images = images[:val_size]
    train_images = images[val_size:]

    # Copy train images
    for img in train_images:
        src = os.path.join(train_class_path, img)
        dst = os.path.join(OUTPUT_DIR, "train", new_class, img)
        shutil.copyfile(src, dst)

    # Copy val images
    for img in val_images:
        src = os.path.join(train_class_path, img)
        dst = os.path.join(OUTPUT_DIR, "val", new_class, img)
        shutil.copyfile(src, dst)

# --- Handle Test Split (Keep as is)
for original_class, new_class in class_mapping.items():
    test_class_path = os.path.join(SOURCE_DIR, "test", original_class)
    images = os.listdir(test_class_path)

    for img in images:
        src = os.path.join(test_class_path, img)
        dst = os.path.join(OUTPUT_DIR, "test", new_class, img)
        shutil.copyfile(src, dst)

print("Dataset prepared successfully.")

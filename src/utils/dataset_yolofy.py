import json
import os
import random
from shutil import move

# Load the JSON data
with open("nightowls_v/nightowls_validation.json", "r") as file:
    data = json.load(file)

# Create directory structure
dirs = [
    "train/images", "train/labels",
    "val/images", "val/labels",
    "test/images", "test/labels"
]

for d in dirs:
    os.makedirs(f"nightowls_v/{d}", exist_ok=True)

# Split the dataset
image_count = len(data["images"])
train_count = int(0.85 * image_count)
val_count = int(0.10 * image_count)
test_count = image_count - train_count - val_count

image_files = [img["file_name"] for img in data["images"]]
random.shuffle(image_files)

train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Map from category ID to its zero-indexed ID
category_mapping = {category["id"]: idx for idx, category in enumerate(data["categories"]) if category["name"] != "ignore"}

# Helper function to write annotations in YOLO format
def write_annotations(files, split):
    for file in files:
        # Extract image info
        image_info = [img for img in data["images"] if img["file_name"] == file][0]
        
        # Get annotations for this image
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == image_info["id"] and ann["category_id"] in category_mapping.keys()]
        
        # Create a .txt file for the labels
        with open(f"nightowls_v/{split}/labels/{file.replace('.png', '.txt')}", "w") as label_file:
            for ann in annotations:
                bbox = ann["bbox"]
                x_center = (bbox[0] + bbox[2] / 2) / image_info["width"]
                y_center = (bbox[1] + bbox[3] / 2) / image_info["height"]
                width = bbox[2] / image_info["width"]
                height = bbox[3] / image_info["height"]
                
                category_id = category_mapping[ann["category_id"]]
                
                label_file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")
        
        # Move the image to its appropriate folder
        move(f"nightowls_v/images/{file}", f"nightowls_v/{split}/images/{file}")

# Write annotations and move files
write_annotations(train_files, "train")
write_annotations(val_files, "val")
write_annotations(test_files, "test")

# Generate the data.yaml file
names = [category["name"] for category in data["categories"] if category["name"] != "ignore"]

with open("data.yaml", "w") as yaml_file:
    yaml_file.write("train: nightowls_v/train/images\n")
    yaml_file.write("val: nightowls_v/val/images\n")
    yaml_file.write("test: nightowls_v/test/images\n")
    yaml_file.write(f"nc: {len(names)}\n")
    yaml_file.write(f"names: {names}\n")

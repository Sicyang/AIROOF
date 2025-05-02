import os
import shutil
import random
import re

# Extract camera name
def extract_camera_name(filename):
    match = re.search(r'(Camera\d+)', filename)
    if match:
        return match.group(1)
    if filename.startswith("B0"):
        return filename.split("-")[0]
    return None

def split_by_camera_individually(source_root, dest_root, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    # Collect all images and group them by camera name
    camera_image_map = {}
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                camera_name = extract_camera_name(file)
                if not camera_name:
                    print(f"⚠️ Camera name not recognized, skipping: {file}")
                    continue
                full_path = os.path.join(root, file)
                camera_image_map.setdefault(camera_name, []).append(full_path)

    # Create destination folders for each split
    for split in ['train', 'test', 'val']:
        os.makedirs(os.path.join(dest_root, split), exist_ok=True)

    # Split images for each camera individually
    for camera_name, image_paths in camera_image_map.items():
        random.shuffle(image_paths)
        total = len(image_paths)
        train_end = int(total * train_ratio)
        test_end = train_end + int(total * test_ratio)

        splits = {
            'train': image_paths[:train_end],
            'test': image_paths[train_end:test_end],
            'val': image_paths[test_end:]
        }

        for split_name, files in splits.items():
            for file in files:
                filename = os.path.basename(file)
                dst = os.path.join(dest_root, split_name, filename)
                shutil.copy2(file, dst)

        print(f"📦 {camera_name} ➜ train: {len(splits['train'])}, test: {len(splits['test'])}, val: {len(splits['val'])}")

if __name__ == "__main__":
    source_folder = r"C:\Users\yqjys\Desktop\AIroof\sorted_by_camera"
    destination_root = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images"
    split_by_camera_individually(source_folder, destination_root)


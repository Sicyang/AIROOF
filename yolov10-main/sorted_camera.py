import os
import shutil
import re

# Automatically extract camera name
def extract_camera_name(filename):
    match = re.search(r'(Camera\d+)', filename)
    if match:
        return match.group(1)
    if filename.startswith("B0"):
        return filename.split("-")[0]
    return None

def classify_images_by_camera(src_root, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # Traverse all subdirectories
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            camera_name = extract_camera_name(file)
            if not camera_name:
                print(f"⚠️ Unable to recognize camera name, skipping: {file}")
                continue

            # Construct destination folder path
            target_folder = os.path.join(dst_root, camera_name)
            os.makedirs(target_folder, exist_ok=True)

            # Copy file
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_folder, file)
            shutil.copy2(src_path, dst_path)
            print(f"✅ Categorized: {file} ➜ {camera_name}")

if __name__ == "__main__":
    # Get current script directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths
    source_folder = os.path.join(base_dir, "..", "sorted_by_camera")
    destination_root = os.path.join(base_dir, "datasets", "images")

    classify_images_by_camera(source_folder, destination_root)

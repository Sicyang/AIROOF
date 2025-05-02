import cv2
import numpy as np
import json
import re
import os
from PIL import Image

# === Ensure the PNG file is valid and fix it if necessary ===
def ensure_valid_png(image_path):
    try:
        # Try to open the image with PIL; if successful, it's at least decodable
        img = Image.open(image_path)
        img = img.convert("RGB")  # Remove alpha channel if present
        fixed_path = os.path.splitext(image_path)[0] + "_fixed.png"
        img.save(fixed_path, "PNG")
        print(f"[FIXED] Saved valid PNG: {fixed_path}")

        # Delete the original and replace it with the fixed PNG
        os.remove(image_path)
        os.rename(fixed_path, image_path)
        print(f"[CLEANUP] Replaced original with fixed PNG: {image_path}")

        return image_path
    except Exception as e:
        print(f"[ERROR] Invalid PNG: {image_path}, error: {e}")
        return None

# === Automatically extract camera name from filename ===
def extract_camera_name(filename):
    match = re.search(r'(Camera\d+)', filename)
    if match:
        return match.group(1)
    if filename.startswith("B0"):
        return filename.split("-")[0]
    return None

# === Load camera configuration from JSON file ===
def load_camera_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === Find the corresponding configuration based on the image filename ===
def get_camera_config(img_path, config_dict):
    import os

    img_name = os.path.basename(img_path)

    # === Automatically recognize camera name ===
    if img_name.startswith("B000"):
        # Example: B000C00U2513_20240101.png
        camera_name = img_name.split("-")[0]
    elif "Camera" in img_name:
        # Example: Camera3_20240101.jpg
        camera_name = img_name.split("_")[0]
    elif img_name.startswith("B0"):
        # Example: B050C01U2501-8680xxxxxx.jpg
        camera_name = img_name.split("-")[0]
    else:
        raise ValueError(f"❌ Unable to recognize camera name: {img_name}")

    # === Check if the camera name exists in the configuration ===
    if camera_name not in config_dict:
        raise ValueError(f"❌ Camera {camera_name} not found in configuration")

    # === Get the configuration and add the camera name field ===
    config = config_dict[camera_name]
    config["camera_name"] = camera_name  # ✅ Add camera name for TXT output

    return config

# === Convert the image to 8-bit format ===
def convert_to_8bit(image):
    if image is None:
        raise ValueError("Image is None, possibly due to an incorrect path")
    if image.dtype == np.uint16:
        return (image / 256).astype('uint8')
    return image.astype('uint8')

# === Main image processing function ===
def process_image(image_path, config_dict):
    print(f"Processing image: {image_path}")
    config = get_camera_config(image_path, config_dict)
    reference_img = cv2.imread(config["reference_image"])
    reference_img = convert_to_8bit(reference_img)
    ref_height, ref_width = reference_img.shape[:2]

    # Create a blank mask with the same size as the reference image
    mask = np.zeros((ref_height, ref_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(config["roi_points"], dtype=np.int32)], 255)

    # Ensure the image is a valid PNG (optional fix)
    if image_path.lower().endswith(".png"):
        fixed_path = ensure_valid_png(image_path)
        if fixed_path and os.path.exists(fixed_path):
            image_path = fixed_path

    img = cv2.imread(image_path)
    img = convert_to_8bit(img)
    resized_img = cv2.resize(img, (ref_width, ref_height))

    # Apply the ROI mask
    roi_img = np.zeros_like(resized_img)
    roi_img[mask == 255] = resized_img[mask == 255]

    # Histogram equalization to enhance contrast in the ROI
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    equalized_roi = cv2.equalizeHist(gray_roi)
    final_img = cv2.cvtColor(equalized_roi, cv2.COLOR_GRAY2BGR)
    roi_img[mask == 255] = final_img[mask == 255]

    return roi_img

# === Apply processing to all images in a folder and overwrite them ===
def apply_processing_and_overwrite(folder_path, config_dict):
    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(folder_path, file)
        try:
            processed = process_image(image_path, config_dict)
            cv2.imwrite(image_path, processed)
            print(f"✅ Overwritten: {file}")
        except Exception as e:
            print(f"❌ Skipped {file}, error: {e}")

# === Main entry point ===
if __name__ == "__main__":
    # Path to JSON configuration file
    config_path = r"C:\Users\yqjys\Desktop\AIroof\camera_config.json"
    config_dict = load_camera_config(config_path)

    # Path to the dataset
    base_path = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images"
    for subset in ["train", "test", "val"]:
        folder = os.path.join(base_path, subset)
        apply_processing_and_overwrite(folder, config_dict)

import cv2
import json
import os
import re

# Function to automatically extract camera name
def extract_camera_name(filename):
    match = re.search(r'(Camera\d+)', filename)
    if match:
        return match.group(1)
    if filename.startswith("B0"):
        return filename.split("-")[0]
    return None

# Save configuration, with added reference_image
def save_camera_config(camera_name, roi_points, reference_image_path, save_path):
    config = {
        camera_name: {
            "model_path_detect": r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect\train\weights\best.pt",
            "model_path_segment": r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\segment\train\weights\best.pt",
            "reference_image": reference_image_path.replace("\\", "/"),  # Use forward slashes for compatibility
            "roi_points": roi_points,
            "confidence_threshold": 0.2,
            "area_ratio_threshold": 0.1
        }
    }

    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            all_config = json.load(f)
    else:
        all_config = {}

    all_config.update(config)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_config, f, indent=4)
    print(f"\n✅ Configuration saved: {camera_name} -> {save_path}\n")

# Main logic
def main():
    image_folder = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\standard"
    config_save_path = r"C:\Users\yqjys\Desktop\AIroof\camera_config.json"
    scale = 0.5

    # Load existing configuration if available
    if os.path.exists(config_save_path):
        with open(config_save_path, 'r', encoding='utf-8') as f:
            existing_config = json.load(f)
    else:
        existing_config = {}

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Unable to read image: {filename}")
            continue

        # Automatically extract camera name
        camera_name = extract_camera_name(filename)
        if not camera_name:
            print(f"⚠️ Camera name not recognized, skipping: {filename}")
            continue

        if camera_name in existing_config:
            print(f"✅ Already configured, skipping: {camera_name}")
            continue

        print(f"\n🖼️ Current image: {filename}  ➜  Camera: {camera_name}")
        img_resized = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("Select ROI", img_resized, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            print("⚠️ No region selected, skipping")
            continue

        x, y, w, h = roi
        x_ori, y_ori = int(x / scale), int(y / scale)
        w_ori, h_ori = int(w / scale), int(h / scale)

        roi_points = [
            [x_ori, y_ori],
            [x_ori + w_ori, y_ori],
            [x_ori + w_ori, y_ori + h_ori],
            [x_ori, y_ori + h_ori]
        ]

        print("\n📌 ROI region (original image coordinates):")
        for pt in roi_points:
            print(tuple(pt))

        # Save config with reference_image field
        save_camera_config(camera_name, roi_points, image_path, config_save_path)
        existing_config[camera_name] = True  # Prevent duplicate configuration

if __name__ == "__main__":
    main()

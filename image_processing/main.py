import os
import cv2
from image_processing import (
    process_image,
    convert_to_8bit,
    get_camera_config
)

# 1. 设置相对路径
BASE_DIR = os.path.normpath(r'.\yolov10-main\datasets\images')
IMAGE_DIR = os.path.join(BASE_DIR, "test")  # 图片文件夹
OUTPUT_DIR = os.path.join(IMAGE_DIR, "processed_images")  # 处理后图片的保存目录

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 处理文件夹中的所有图片
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    output_path = os.path.join(OUTPUT_DIR, image_file)

    print(f"Processing image: {image_path}")

    # 读取并转换图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}, skipping...")
        continue

    image = convert_to_8bit(image)

    # 识别相机配置
    try:
        config = get_camera_config(image_path)
        print(f"Camera configuration loaded: {config}")
    except ValueError as e:
        print(e)

    # 运行图像处理
    processed_img = process_image(image_path)

    # 保存处理后的图片
    cv2.imwrite(output_path, processed_img)
    print(f"Processed image saved to: {output_path}")

print("Image processing pipeline completed.")


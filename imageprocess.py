import os
import cv2
import numpy as np
from OpenCV import process_image  # Assuming process_image is the function used

def process_directory(directory):
    """
    Process all images in the given directory using the predefined process_image function.
    """
    processed_dir = os.path.join(directory, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('png', 'jpg', 'jpeg')):
            try:
                processed_img = process_image(file_path)
                processed_img_path = os.path.join(processed_dir, file_name)
                cv2.imwrite(processed_img_path, processed_img)
                print(f"Processed and saved: {processed_img_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    directories = [
        r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\train',
        r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\test',
        r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\val'
    ]

    for directory in directories:
        process_directory(directory)

    print("Image processing completed for all directories.")
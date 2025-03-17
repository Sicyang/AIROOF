import cv2
import numpy as np
from .config import camera_configs

def convert_to_8bit(image):
    """Convert the image to 8-bit if it is not already in that format."""
    if image.dtype == np.uint16:
        return (image / 256).astype('uint8')
    return image.astype('uint8')

def get_camera_config(image_path):
    """Identify the camera configuration based on the image name."""
    for cam, config in camera_configs.items():
        if cam in image_path.lower():
            return config
    raise ValueError("Unknown camera source in image name")

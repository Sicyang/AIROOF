# image_training/train.py
import os
from ultralytics import YOLO

# 设定相对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.normpath(r'.\yolov10-main\models\yolov8n-seg.yaml')
WEIGHT_PATH = os.path.normpath(r'.\yolov10-main\models\yolov8n-seg.pt')
DATA_CONFIG = os.path.normpath(r'.\yolov10-main\models\data.yaml')

def load_yolo_model(config_path=None, weight_path=None):
    """
    Load a YOLO model for training.

    Args:
        config_path (str, optional): Path to YOLO model configuration file (.yaml).
        weight_path (str, optional): Path to trained model weights (.pt).

    Returns:
        YOLO model instance
    """
    if weight_path:
        return YOLO(weight_path)
    elif config_path:
        return YOLO(config_path)
    else:
        raise ValueError("Either config_path or weight_path must be provided.")

def train_yolo_model(model, data_config, epochs=500, img_size=640):
    """
    Train a YOLO model with the specified dataset.

    Args:
        model (YOLO): Loaded YOLO model.
        data_config (str): Path to dataset configuration file (.yaml).
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
    """
    model.train(data=data_config, epochs=epochs, imgsz=img_size)

if __name__ == "__main__":
    model = load_yolo_model(CONFIG_PATH, WEIGHT_PATH)
    train_yolo_model(model, DATA_CONFIG)
    print("Training complete.")


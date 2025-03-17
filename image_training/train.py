# image_training/train.py
from ultralytics import YOLO

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
    config_path = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\yolov8n-seg.yaml"
    weight_path = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\yolov8n-seg.pt"
    data_config = r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\data.yaml"

    model = load_yolo_model(config_path, weight_path)
    train_yolo_model(model, data_config)
    print("Training complete.")



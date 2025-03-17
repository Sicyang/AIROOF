from image_training import (
    load_yolo_model,
    train_yolo_model
)

# 1. 设置参数
CONFIG_PATH = "models/yolov8n-seg.yaml"  # YOLO 配置文件
WEIGHT_PATH = "models/yolov8n-seg.pt"  # 预训练模型
DATA_CONFIG = "models/data.yaml"  # 数据集配置
EPOCHS = 100
IMG_SIZE = 640

# 2. 加载 YOLO 模型
print("Loading YOLO model for training...")
model = load_yolo_model(CONFIG_PATH, WEIGHT_PATH)

# 3. 训练模型
print("Starting YOLO training...")
train_yolo_model(model, DATA_CONFIG, epochs=EPOCHS, img_size=IMG_SIZE)

print("YOLO training completed.")

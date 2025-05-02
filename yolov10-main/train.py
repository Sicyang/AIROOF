from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\yolov8.yaml")
    model = YOLO(r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\yolov8n.pt")

    model.train(data=r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\models\data.yaml', epochs=500, imgsz=640)


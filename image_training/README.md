# Image Training Module

## Overview
This module handles the training and inference of a YOLO-based object detection model. It includes functionalities for training a model, running inference on test images, and configuring training parameters.

## File Descriptions

### 1. `detect.py`
- Loads a trained YOLO model and runs object detection on images.
- Allows setting a confidence threshold for detections.
- Usage:
  ```python
  model = load_yolo_model("path/to/model.pt")
  results = run_detection(model, "path/to/image.jpg")
  ```

### 2. `train.py`
- Loads and trains a YOLO model on a given dataset.
- Requires either a model configuration file (`.yaml`) or pretrained weights (`.pt`).
- Trains the model using a specified dataset.
- Usage:
  ```python
  model = load_yolo_model("path/to/config.yaml", "path/to/weights.pt")
  train_yolo_model(model, "path/to/data.yaml", epochs=500, img_size=640)
  ```

### 3. `main.py`
- Entry point for training.
- Loads a YOLO model and starts training with predefined parameters.
- Configures the dataset, model architecture, and training settings.
- Usage:
  ```bash
  python main.py
  ```

## Dependencies
- `ultralytics` (for YOLO model handling)
- `torch`
- `numpy`
- `opencv-python`

## How Files Work Together
1. `detect.py` is used for running inference on trained models.
2. `train.py` is used to train a YOLO model on a dataset.
3. `main.py` simplifies the training process by setting parameters and running `train.py`.

## Usage Guide
1. Ensure the `ultralytics` package is installed.
2. Train the model using `train.py` or `main.py`.
3. Run `detect.py` to test the trained model on images.
4. Fine-tune the model if necessary by adjusting the training parameters.

This module provides a structured workflow for training and using YOLO models efficiently.


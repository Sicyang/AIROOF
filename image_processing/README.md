# Image Processing Module

## Overview
This module is responsible for processing images captured by multiple cameras. It includes functionalities such as image conversion, annotation transformation, image splitting, and processing.

## File Descriptions

### 1. `config.py`
- Stores camera-specific configurations, including:
  - Reference images for each camera
  - Region of Interest (ROI) points
- Used by other scripts to determine how to process images from different cameras.

### 2. `convert_labels.py`
- Converts JSON annotation files into YOLO TXT format.
- Normalizes coordinates to ensure consistency.
- Usage:
  ```bash
  python convert_labels.py --json-dir path/to/jsons --save-dir path/to/output --classes "leaves“

### 3. `copy_images.py`
- Copies images from multiple source folders to a single destination folder.
- Usage:
  ```python
  copy_camera_images("source_directory", "destination_directory")
  ```

### 4. `process_images.py`
- Processes images by:
  - Resizing
  - Converting to 8-bit
  - Extracting the ROI
  - Applying histogram equalization
- Uses:
  - `get_camera_config()` from `utils.py`
  - `convert_to_8bit()` from `utils.py`
- Usage:
  ```python
  processed_img = process_image("path/to/image.jpg")
  ```

### 5. `select_anchor.py`
- Allows users to manually select anchor points on an image using mouse clicks.
- Displays the image and records selected points.
- Usage:
  ```python
  points = get_anchor_points("path/to/image.jpg")
  ```

### 6. `split_images.py`
- Splits images into training, validation, and test sets.
- Moves images from `source_dir` into `test_dir` and `val_dir` based on predefined ratios.
- Usage:
  ```python
  split_images("source_dir", "test_dir", "val_dir", test_ratio=0.2, val_ratio=0.2)
  ```

### 7. `utils.py`
- Contains helper functions:
  - `convert_to_8bit(image)`: Converts images to 8-bit format.
  - `get_camera_config(image_path)`: Retrieves camera configuration based on the image filename.
- Used in multiple scripts.

### 8. `main.py`
- Entry point of the module.
- Processes all images in a specified directory:
  - Loads camera configurations.
  - Converts images to 8-bit format.
  - Calls `process_image()` to process images.
  - Saves processed images to an output directory.
- Usage:
  ```bash
  python main.py
  ```

## Dependencies
- OpenCV (`cv2`)
- NumPy (`numpy`)
- tqdm
- json
- argparse

## How Files Work Together
1. `config.py` provides camera configurations and ROIs.
2. `utils.py` offers helper functions used across different scripts.
3. `convert_labels.py` processes annotation labels.
4. `copy_images.py` extracts relevant images.
5. `process_images.py` enhances image quality and extracts ROIs.
6. `select_anchor.py` allows manual point selection to decide ROIs.
7. `split_images.py` partitions the dataset for training.
8. `main.py` orchestrates the processing pipeline.

## Usage Guide
1. Ensure all dependencies are installed.
2. Run `copy_images.py` to collect relevant images.
3. Use `split_images.py` to divide the dataset.
4. If needed, manually select anchor points using `select_anchor.py`.
5. Run `main.py` to process images.
6. Use Labelme to label the processed images.
7. Use `convert_labels.py` to convert annotations.

This module is structured to efficiently preprocess and organize images for machine learning tasks.


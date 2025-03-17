# Deployment Module

## Overview
This module handles the deployment of trained YOLO models for object detection. It includes functions for loading trained models, running inference on new images, processing images, and generating reports.

## File Descriptions

### 1. `predict.py`
- Loads YOLO models trained for different cameras.
- Processes images and applies object detection.
- Generates reports in both PDF and TXT formats.
- Merges multiple PDFs into a single report.
- Deletes temporary prediction folders after processing.

### Key Functions:
- `convert_to_png(img_path)`: Converts an image to PNG format if not already in that format.
- `write_to_txt(file_path, data)`: Writes detection results to a TXT file.
- `get_camera_config(img_url)`: Retrieves the camera-specific configuration based on the image filename.
- `process_image_and_generate_report(img_url, pdf_file_list, excel_data)`: Processes an image, applies YOLO inference, and generates a PDF report.
- `merge_pdfs(pdf_file_list, output_pdf_path)`: Merges multiple PDFs into a single report.
- `delete_predict_folders()`: Cleans up temporary YOLO prediction directories.

### Usage:
To run the deployment module:
```bash
python predict.py
```

## Dependencies
- `ultralytics` (for YOLO model handling)
- `opencv-python` (for image processing)
- `numpy`
- `PIL` (for image format conversion)
- `fpdf` (for PDF generation)
- `PyPDF2` (for merging PDFs)

## Workflow
1. Images are processed and converted if needed.
2. YOLO detection is applied based on camera configurations.
3. Results are saved in TXT and PDF format.
4. Multiple PDFs are merged into a final report.
5. Temporary YOLO prediction directories are deleted.

This module enables efficient deployment of the trained YOLO models for automated object detection and report generation.


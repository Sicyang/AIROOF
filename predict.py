import datetime
import os
import shutil
import sys
import cv2
import numpy as np
from fpdf import FPDF
from ultralytics import YOLO
from PIL import Image
from PyPDF2 import PdfMerger
from image_processing.process_images import process_image

# 设置相对路径的根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义相对路径
MODEL_DIR = os.path.normpath(r'.\yolov10-main\runs')
DATASET_DIR = os.path.normpath(r'.\yolov10-main\datasets\images')
RESULTS_DIR = os.path.normpath(r'.\results')
PROCESSED_DIR = os.path.normpath(r'.\yolov10-main\processed')

# 确保必要的文件夹存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Camera configurations
camera_configs = {
    "camera1": {
        "model_path": os.path.normpath(r'.\yolov10-main\runs\detect_camera1\train\weights\best.pt'),
        "roi_points": [(299, 92), (305, 78), (316, 68), (335, 77), (363, 85), (369, 100), (371, 118), (338, 116),
                       (319, 108), (309, 100)],
        "confidence_threshold": 0.7,
        "area_ratio_threshold": 0.1
    },
    "camera2": {
        "model_path": os.path.normpath(r'.\yolov10-main\runs\detect_camera2456\train\weights\best.pt'),
        "roi_points": [(8, 260), (49, 265), (92, 272), (140, 281), (197, 282), (238, 302), (286, 300),
                       (319, 329), (343, 352), (354, 395), (356, 475), (352, 524), (349, 559), (345, 607),
                       (336, 665), (332, 705), (290, 707), (245, 717), (199, 714), (134, 718), (77, 718)],
        "confidence_threshold": 0.7,
        "area_ratio_threshold": 0.1
    }
}

def convert_to_png(img_path):
    """ Convert image to PNG format if it's not already PNG """
    file_name, file_extension = os.path.splitext(img_path)
    new_img_path = file_name + '.png'
    if file_extension.lower() != '.png':
        img = Image.open(img_path)
        img = img.convert("RGB")
        img.save(new_img_path, 'PNG')
        print(f"Converted {img_path} to {new_img_path}")
        return new_img_path
    return img_path

def write_to_txt(file_path, data):
    """ Write data to a text file """
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(", ".join(map(str, data)) + "\n")
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

def get_camera_config(img_url):
    """ Get the camera configuration based on image name """
    for cam, config in camera_configs.items():
        if cam in img_url.lower():
            return cam, config
    raise ValueError("Unknown camera source in image name")

def process_image_and_generate_report(img_url, pdf_file_list, excel_data):
    """ Process an image and generate a report """
    try:
        img_url = convert_to_png(img_url)
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_temp_path = os.path.join(RESULTS_DIR, f"result_{current_time_str}.pdf")
        txt_file_path = os.path.join(RESULTS_DIR, f"results_{current_time_str}.txt")

        camera_type, config = get_camera_config(img_url)
        model = YOLO(config["model_path"])
        roi_points = config["roi_points"]
        confidence_threshold = config["confidence_threshold"]
        area_ratio_threshold = config["area_ratio_threshold"]

        processed_img = process_image(img_url)
        processed_img_path = os.path.join(PROCESSED_DIR, "processed_image.png")
        cv2.imwrite(processed_img_path, processed_img)

        results = model(processed_img_path, save=True)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(200, 10, txt="Original Image:", ln=True)
        pdf.image(img_url, x=10, y=30, w=90)

        retrain_required, has_overlap, area_ratio = False, False, 0
        for result in results:
            for box in result.boxes:
                if box.conf.item() > confidence_threshold:
                    retrain_required = True
                    break

        output_texts = []
        if retrain_required:
            model_seg = YOLO(config["model_path"])
            results_seg = model_seg(processed_img_path, save=True)
            for result in results_seg:
                if result.masks:
                    mask = result.masks.data[0].cpu().numpy()
                    roi_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.fillPoly(roi_mask, [np.array(roi_points, dtype=np.int32)], 1)
                    has_overlap = np.any(np.logical_and(mask, roi_mask))
                    area_ratio = np.sum(mask) / (mask.shape[0] * mask.shape[1])
                    if has_overlap or area_ratio > area_ratio_threshold:
                        output_texts.append("Clean-up required!")
                    else:
                        output_texts.append("No cleaning needed.")

        pdf.cell(200, 10, txt="Inference Results:", ln=True)
        for text in output_texts:
            pdf.cell(200, 10, txt=text, ln=True)
        pdf.output(pdf_temp_path)
        pdf_file_list.append(pdf_temp_path)
        write_to_txt(txt_file_path, [camera_type, "Detected", area_ratio])

    except Exception as e:
        print(f"Error processing {img_url}: {e}")

def merge_pdfs(pdf_file_list, output_pdf_path):
    """ Merge multiple PDF reports into one """
    merger = PdfMerger()
    for pdf in pdf_file_list:
        merger.append(pdf)
    merger.write(output_pdf_path)
    merger.close()
    print(f"Merged PDF saved at {output_pdf_path}")

def delete_predict_folders():
    """ Delete temporary prediction folders """
    for base_dir in [os.path.normpath(r'.\yolov10-main\runs\detect_camera1')]:
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path) and "predict" in folder:
                shutil.rmtree(folder_path)

def main():
    """ Main function to process all images """
    img_dir = os.path.normpath(r'.\running test')
    pdf_file_list, excel_data = [], []

    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image_and_generate_report(img_path, pdf_file_list, excel_data)

    merged_pdf_path = os.path.join(RESULTS_DIR, "merged_results.pdf")
    merge_pdfs(pdf_file_list, merged_pdf_path)
    delete_predict_folders()

if __name__ == "__main__":
    main()

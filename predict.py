import datetime
import os
import shutil
import sys
import cv2
import numpy as np
import pandas as pd
from fpdf import FPDF
from ultralytics import YOLO
from PIL import Image
from PyPDF2 import PdfMerger

# Adding custom library path
sys.path.append(r'C:\Users\yqjys\Desktop\AIroof\yolov10-main')
from OpenCV import process_image  # Assuming process_image is the function used

# Camera configurations
camera_configs = {
    "camera1": {
        "model_path": r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect_camera1\train\weights\best.pt',
        "roi_points": [(299, 92), (305, 78), (316, 68), (335, 77), (363, 85), (369, 100), (371, 118), (338, 116),
                       (319, 108), (309, 100)],
        "confidence_threshold": 0.7,
        "area_ratio_threshold": 0.1
    },
    "camera2": {
        "model_path": r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect_camera2456\train\weights\best.pt',
        "roi_points": [(8, 260), (49, 265), (92, 272), (140, 281), (197, 282), (238, 302), (286, 300),
                       (319, 329), (343, 352), (354, 395), (356, 475), (352, 524), (349, 559), (345, 607),
                       (336, 665), (332, 705), (290, 707), (245, 717), (199, 714), (134, 718), (77, 718),
                       (31, 717), (10, 702), (0, 659), (2, 615), (2, 582), (0, 545), (0, 511), (2, 472),
                       (4, 436), (6, 397), (5, 363), (2, 325), (0, 289)],
        "confidence_threshold": 0.7,
        "area_ratio_threshold": 0.1
    },
    "camera3": {
        "model_path": r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect_camera3\train\weights\best.pt',
        "roi_points": [(591, 480), (542, 445), (517, 429), (495, 397), (473, 376), (450, 344), (419, 314), (386, 284),
                       (371, 263), (352, 233), (330, 215), (302, 183), (281, 175), (260, 185), (264, 209), (285, 240),
                       (304, 263), (322, 285), (339, 308), (364, 336), (383, 357), (400, 381), (416, 392), (435, 413),
                       (449, 432), (469, 452), (496, 484), (517, 507), (542, 529), (568, 511), (588, 492)],
        "confidence_threshold": 0.7,
        "area_ratio_threshold": 0.1
    },
    "camera4": {
        "model_path": r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect_camera2456\train\weights\best.pt',
        "roi_points": [(304, 56), (308, 130), (300, 180), (305, 253), (302, 311), (313, 343), (355, 342), (410, 352),
                       (454, 350), (533, 353),
                       (600, 351), (637, 344), (628, 282), (628, 235), (629, 190), (629, 117), (628, 80), (627, 43),
                       (606, 27), (547, 17),
                       (491, 26), (422, 22), (360, 42), (313, 45)],
        "confidence_threshold": 0.3,
        "area_ratio_threshold": 0.01
    },
    "camera5": {
        "model_path": r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect_camera2456\train\weights\best.pt',
        "roi_points": [(373, 591), (378, 488), (383, 443), (380, 381), (382, 321), (378, 260), (391, 190),
                       (392, 131), (394, 81), (303, 66), (198, 50), (123, 41), (62, 31), (39, 67),
                       (28, 125), (22, 214), (9, 278), (4, 335), (2, 417), (6, 482), (1, 539), (5, 574),
                       (12, 623), (97, 630), (201, 626), (289, 623), (358, 616)],
        "confidence_threshold": 0.7,
        "area_ratio_threshold": 0.1
    },
    "camera6": {
        "model_path": r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect_camera2456\train\weights\best.pt',
        "roi_points": [(30, 31), (100, 43), (189, 40), (237, 36), (321, 47), (400, 52), (472, 63),
                       (551, 82), (626, 143), (630, 214), (649, 313), (669, 445), (652, 384),
                       (664, 514), (658, 575), (645, 617), (578, 656), (478, 690), (367, 699),
                       (278, 700), (171, 693), (81, 677), (44, 607), (40, 524), (27, 445),
                       (16, 375), (11, 315), (17, 251), (21, 176), (25, 91)],
        "confidence_threshold": 0.7,
        "area_ratio_threshold": 0.1
    },
}


def convert_to_png(img_path):
    file_name, file_extension = os.path.splitext(img_path)
    new_img_path = file_name + '.png'
    if file_extension.lower() != '.png':
        img = Image.open(img_path)
        img = img.convert("RGB")  # 确保是RGB模式，否则可能有透明通道问题
        img.save(new_img_path, 'PNG')
        print(f"Converted {img_path} to {new_img_path}")
        return new_img_path
    return img_path



def write_to_txt(file_path, data):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(", ".join(map(str, data)) + "\n")
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def get_camera_config(img_url):
    for cam, config in camera_configs.items():
        if cam in img_url.lower():
            return cam, config
    raise ValueError("Unknown camera source in image name")


def process_image_and_generate_report(img_url, pdf_file_list, excel_data):
    try:
        img_url = convert_to_png(img_url)
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_temp_path = fr"C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\result_{current_time_str}.pdf"
        txt_file_path = fr"C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\results_{current_time_str}.txt"

        camera_type, config = get_camera_config(img_url)
        model = YOLO(config["model_path"])
        roi_points = config["roi_points"]
        confidence_threshold = config["confidence_threshold"]
        area_ratio_threshold = config["area_ratio_threshold"]

        processed_img = process_image(img_url)
        processed_img_path = fr"C:\Users\yqjys\Desktop\AIroof\yolov10-main\processed\processed_image.png"
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
            model_seg = YOLO(config["model_path"])  # Using the same model for segmentation
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
    merger = PdfMerger()
    for pdf in pdf_file_list:
        merger.append(pdf)
    merger.write(output_pdf_path)
    merger.close()
    print(f"Merged PDF saved at {output_pdf_path}")


def delete_predict_folders():
    for base_dir in [r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect"]:
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path) and "predict" in folder:
                shutil.rmtree(folder_path)

def main():
    img_dir = r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\running test'
    pdf_file_list, excel_data = [], []
    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image_and_generate_report(img_path, pdf_file_list, excel_data)
    merge_pdfs(pdf_file_list, r'C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\merged_results.pdf')
    delete_predict_folders()


if __name__ == "__main__":
    main()
import datetime
import os
import shutil
import sys
import cv2
import numpy as np
import json
from fpdf import FPDF
from ultralytics import YOLO
from PIL import Image
from PyPDF2 import PdfMerger

# Add yolov10-main to module path
sys.path.append(r'C:\Users\yqjys\Desktop\AIroof\yolov10-main')

# ✅ Import custom OpenCV processing functions
from OpenCV import process_image, load_camera_config, get_camera_config

# === Determine the warning level based on area ratio and overlap ===
def get_warning_level(area_ratio, has_overlap, threshold):
    if area_ratio > threshold * 2:
        return "Red", "RED WARNING: Heavy leaf accumulation. Immediate cleaning required!"
    elif area_ratio > threshold:
        return "Yellow", "YELLOW WARNING: Leaf accumulation detected. Please check soon."
    elif has_overlap:
        return "Yellow", "YELLOW WARNING: Leaves detected near the drainage area. Please be alert."
    else:
        return "Green", "GREEN: No cleaning needed."

# === Convert any image to PNG format ===
def convert_to_png(img_path):
    file_name, file_extension = os.path.splitext(img_path)
    new_img_path = file_name + '.png'
    try:
        img = Image.open(img_path)
        if img.mode == 'I;16' or '16' in str(img.mode) or np.array(img).dtype == np.uint16:
            print(f"[INFO] Converting 16-bit image to 8-bit: {img_path}")
            img = img.convert("RGB")  # Force conversion to 8-bit 3-channel image
        else:
            img = img.convert("RGB")  # Ensure no alpha channel
        img.save(new_img_path, 'PNG')
        print(f"Converted {img_path} to {new_img_path}")
        return new_img_path
    except Exception as e:
        print(f"[ERROR] Failed to convert {img_path} to PNG: {e}")
        return img_path  # fallback

# === Append detection results to a TXT file ===
def write_to_txt(file_path, data):
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(", ".join(map(str, data)) + "\n")
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

# === Perform image detection, segmentation, and generate a report ===
def process_image_and_generate_report(img_url, pdf_file_list, excel_data, config_dict):
    try:
        print(f"\nProcessing image: {img_url}")
        img_url = convert_to_png(img_url)
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_temp_path = fr"C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\result_{current_time_str}.pdf"
        txt_file_path = fr"C:\Users\yqjys\Desktop\AIroof\yolov10-main\results\results_{current_time_str}.txt"

        # ✅ Load camera configuration
        try:
            config = get_camera_config(img_url, config_dict)
        except ValueError as e:
            print(e)
            return

        model = YOLO(config["model_path_detect"])
        roi_points = config["roi_points"]
        confidence_threshold = config["confidence_threshold"]
        area_ratio_threshold = config["area_ratio_threshold"]

        # ✅ Preprocess image (crop ROI)
        processed_img = process_image(img_url, config_dict)
        processed_img_path = fr"C:\Users\yqjys\Desktop\AIroof\yolov10-main\processed\processed_image.png"
        cv2.imwrite(processed_img_path, processed_img)

        # ✅ Run detection
        results = model(processed_img_path, save=True)

        # === Initialize PDF ===
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Current Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

        retrain_required, has_overlap = False, False
        area_ratio, roi_area, leaf_area = 0, 0, 0  # Initialize metrics

        for result in results:
            for box in result.boxes:
                if box.conf.item() > confidence_threshold:
                    retrain_required = True
                    break

        output_texts = []
        warning_level, warning_msg = "Green", "GREEN: No cleaning needed."

        # ✅ Segmentation analysis
        if retrain_required:
            model_seg = YOLO(config["model_path_segment"])
            results_seg = model_seg(processed_img_path, save=True)
            for result in results_seg:
                if result.masks:
                    for i, mask_tensor in enumerate(result.masks.data):
                        mask = mask_tensor.cpu().numpy()
                        binary_mask = (mask > 0.5).astype(np.uint8)

                        # === Create ROI mask based on the original processed image size, resize to match segmentation size
                        processed_img_cv = cv2.imread(processed_img_path)
                        roi_mask = np.zeros(processed_img_cv.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(roi_mask, [np.array(roi_points, dtype=np.int32)], 1)
                        roi_mask_resized = cv2.resize(roi_mask, (binary_mask.shape[1], binary_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

                        has_overlap = np.any(np.logical_and(binary_mask, roi_mask_resized))
                        binary_mask_roi = np.logical_and(binary_mask, roi_mask_resized)

                        roi_area = np.count_nonzero(roi_mask_resized)
                        leaf_area = np.sum(binary_mask_roi)

                        if roi_area > 0:
                            ratio = leaf_area / roi_area
                            if ratio > area_ratio:  # Keep the largest area ratio found
                                area_ratio = ratio
                                warning_level, warning_msg = get_warning_level(ratio, has_overlap, area_ratio_threshold)

        output_texts.append(warning_msg)

        # === Insert image title ===
        pdf.cell(200, 10, txt="Original Image:", ln=True)

        # === Insert image ===
        image_x = 10
        image_y = pdf.get_y() + 5
        image_w = 120
        try:
            pdf.image(img_url, x=image_x, y=image_y, w=image_w)
        except Exception as e:
            print(f"[WARNING] Skipping image in PDF due to error: {e}")
            pdf.cell(200, 10, txt=f"[Image could not be loaded: {e}]", ln=True)

        # === Set the position for inference text ===
        image_h = image_w * 0.75
        pdf.set_y(image_y + image_h + 10)

        # === Output inference results ===
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Inference Results:", ln=True)
        for text in output_texts:
            pdf.multi_cell(0, 10, txt=text)

        # ✅ Save the PDF
        pdf.output(pdf_temp_path)
        pdf_file_list.append(pdf_temp_path)

        # ✅ Write detection summary to TXT file
        write_to_txt(
            txt_file_path,
            [
                config.get("camera_name", "Unknown"),
                warning_level,
                f"{area_ratio:.4f}",
                roi_area,
                leaf_area
            ]
        )

    except Exception as e:
        print(f"[ERROR] Unexpected failure in processing {img_url}: {e}")

# === Merge multiple PDFs into one ===
def merge_pdfs(pdf_file_list, output_pdf_path):
    merger = PdfMerger()
    for pdf in pdf_file_list:
        merger.append(pdf)
    merger.write(output_pdf_path)
    merger.close()
    print(f"Merged PDF saved at {output_pdf_path}")

# === Delete YOLO detection result folders ===
def delete_predict_folders():
    for base_dir in [r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\detect"]:
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path) and "predict" in folder:
                shutil.rmtree(folder_path)

# === Delete YOLO segmentation result folders ===
def delete_segment_folders():
    for base_dir in [r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\runs\segment"]:
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path) and "predict" in folder:
                shutil.rmtree(folder_path)

# === Generate a legend PDF describing the warning levels ===
def generate_legend_pdf(path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Legend", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="GREEN: No cleaning needed.", ln=True)
    pdf.cell(200, 10, txt="YELLOW WARNING: Leaf accumulation detected. Please check soon.", ln=True)
    pdf.cell(200, 10, txt="RED WARNING: Heavy leaf accumulation. Immediate cleaning required!", ln=True)
    pdf.output(path)

# === Main function ===
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.abspath(os.path.join(current_dir, '..', 'camera_config.json'))
    config_dict = load_camera_config(config_path)

    img_dir = os.path.join(current_dir, 'datasets', 'running test')
    pdf_file_list, excel_data = [], []

    legend_path = os.path.join(current_dir, 'results', 'legend.pdf')
    generate_legend_pdf(legend_path)
    pdf_file_list.append(legend_path)

    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image_and_generate_report(img_path, pdf_file_list, excel_data, config_dict)

    merge_pdfs(pdf_file_list, os.path.join(current_dir, 'results', 'merged_results.pdf'))
    # delete_predict_folders()
    # delete_segment_folders()

if __name__ == "__main__":
    main()

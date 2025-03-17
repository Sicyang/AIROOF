import cv2
import numpy as np
from .utils import convert_to_8bit, get_camera_config

def process_image(image_path):
    """
    Process the image: resize it, convert to 8-bit, extract the ROI, and apply histogram equalization.
    """
    print(f"Processing image: {image_path}")
    config = get_camera_config(image_path)

    # 读取参考图片并转换为 8-bit
    reference_img = cv2.imread(config["reference_image"])
    reference_img = convert_to_8bit(reference_img)
    ref_height, ref_width = reference_img.shape[:2]

    # 创建 ROI mask
    mask = np.zeros((ref_height, ref_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(config["roi_points"], dtype=np.int32)], 255)

    # 读取输入图片
    img = cv2.imread(image_path)
    img = convert_to_8bit(img)
    resized_img = cv2.resize(img, (ref_width, ref_height))

    # 提取 ROI
    roi_img = np.zeros_like(resized_img)
    roi_img[mask == 255] = resized_img[mask == 255]

    # 转换为灰度图并直方图均衡化
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    equalized_roi = cv2.equalizeHist(gray_roi)

    # 转换回 BGR 并应用 ROI
    final_img = cv2.cvtColor(equalized_roi, cv2.COLOR_GRAY2BGR)
    roi_img[mask == 255] = final_img[mask == 255]

    return roi_img
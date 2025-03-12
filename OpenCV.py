import cv2
import numpy as np
import os

# Camera configurations
camera_configs = {
    "camera1": {
        "reference_image": r"C:\\Users\\yqjys\\Desktop\\AIroof\\yolov10-main\\datasets\\images\\standard\\Camera1_April_12_2024_11_35_00_43.png",
        "roi_points": [(83, 358), (94, 337), (113, 312), (138, 273), (161, 235), (178, 212), (201, 176),
                       (219, 153), (243, 125), (267, 97), (288, 78), (308, 62), (322, 52), (331, 63),
                       (350, 73), (367, 79), (374, 99), (383, 133), (390, 165), (396, 185), (404, 219),
                       (416, 253), (425, 290), (432, 320), (441, 359)]
    },
    "camera2": {
        "reference_image": r"C:\Users\yqjys\Desktop\AIroof\yolov10-main\datasets\images\standard\Camera2_May_22_2024_09_34_00_48.png",
        "roi_points": [(8, 260), (49, 265), (92, 272), (140, 281), (197, 282), (238, 302), (286, 300),
                       (319, 329), (343, 352), (354, 395), (356, 475), (352, 524), (349, 559), (345, 607),
                       (336, 665), (332, 705), (290, 707), (245, 717), (199, 714), (134, 718), (77, 718),
                       (31, 717), (10, 702), (0, 659), (2, 615), (2, 582), (0, 545), (0, 511), (2, 472),
                       (4, 436), (6, 397), (5, 363), (2, 325), (0, 289)]
    },
    "camera3": {
        "reference_image": r"C:\\Users\\yqjys\\Desktop\\AIroof\\yolov10-main\\datasets\\images\\standard\\Camera3_May_16_2024_09_20_00_33.png",
        "roi_points": [(591, 480), (542, 445), (517, 429), (495, 397), (473, 376), (450, 344), (419, 314), (386, 284),
                       (371, 263), (352, 233), (330, 215), (302, 183), (281, 175), (260, 185), (264, 209), (285, 240),
                       (304, 263), (322, 285), (339, 308), (364, 336), (383, 357), (400, 381), (416, 392), (435, 413),
                       (449, 432), (469, 452), (496, 484), (517, 507), (542, 529), (568, 511), (588, 492)]
    },
    "camera4": {
        "reference_image": r"C:\\Users\\yqjys\\Desktop\\AIroof\\yolov10-main\\datasets\\images\\standard\\Camera4_June_24_2024_09_51_00_41.png",
        "roi_points": [(528, 134), (520, 208), (514, 291), (506, 367), (503, 440), (484, 536), (484, 576),
                       (517, 614), (568, 637), (624, 649), (735, 679), (760, 700), (815, 708), (878, 715),
                       (942, 718), (1002, 711), (1072, 712), (1153, 714), (1198, 712), (1251, 710), (1277, 711),
                       (1269, 643), (1268, 592), (1268, 521), (1276, 481), (1274, 432), (1272, 397), (1272, 333),
                       (1273, 274), (1278, 201), (1273, 146), (1272, 74), (1245, 26), (1171, 16), (1101, 11),
                       (1009, 16), (937, 10), (855, 17), (783, 22), (689, 50), (608, 80), (538, 108), (575, 97)]
    },
    "camera5": {
        "reference_image": r"C:\\Users\\yqjys\\Desktop\\AIroof\\yolov10-main\\datasets\\images\\standard\\Camera5_June_26_2024_09_45_00_06.png",
        "roi_points": [(373, 591), (378, 488), (383, 443), (380, 381), (382, 321), (378, 260), (391, 190),
                       (392, 131), (394, 81), (303, 66), (198, 50), (123, 41), (62, 31), (39, 67),
                       (28, 125), (22, 214), (9, 278), (4, 335), (2, 417), (6, 482), (1, 539), (5, 574),
                       (12, 623), (97, 630), (201, 626), (289, 623), (358, 616)]
    },
    "camera6": {
        "reference_image": r"C:\\Users\\yqjys\\Desktop\\AIroof\\yolov10-main\\datasets\\images\\standard\\Camera6_June_26_2024_11_39_00_26.png",
        "roi_points": [(30, 31), (100, 43), (189, 40), (237, 36), (321, 47), (400, 52), (472, 63),
                       (551, 82), (626, 143), (630, 214), (649, 313), (669, 445), (652, 384),
                       (664, 514), (658, 575), (645, 617), (578, 656), (478, 690), (367, 699),
                       (278, 700), (171, 693), (81, 677), (44, 607), (40, 524), (27, 445),
                       (16, 375), (11, 315), (17, 251), (21, 176), (25, 91)]
    }
}


def convert_to_8bit(image):
    """
    Convert the image to 8-bit if it is not already in that format.
    """
    if image.dtype == np.uint16:
        return (image / 256).astype('uint8')
    return image.astype('uint8')


def get_camera_config(image_path):
    """
    Identify the camera configuration based on the image name.
    """
    for cam, config in camera_configs.items():
        if cam in image_path.lower():
            return config
    raise ValueError("Unknown camera source in image name")


def process_image(image_path):
    """
    Process the image: resize it, convert to 8-bit, extract the ROI, and apply histogram equalization.
    """
    print(f"Processing image: {image_path}")
    config = get_camera_config(image_path)
    reference_img = cv2.imread(config["reference_image"])
    reference_img = convert_to_8bit(reference_img)
    ref_height, ref_width = reference_img.shape[:2]

    mask = np.zeros((ref_height, ref_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(config["roi_points"], dtype=np.int32)], 255)

    img = cv2.imread(image_path)
    img = convert_to_8bit(img)
    resized_img = cv2.resize(img, (ref_width, ref_height))
    roi_img = np.zeros_like(resized_img)
    roi_img[mask == 255] = resized_img[mask == 255]

    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    equalized_roi = cv2.equalizeHist(gray_roi)
    final_img = cv2.cvtColor(equalized_roi, cv2.COLOR_GRAY2BGR)
    roi_img[mask == 255] = final_img[mask == 255]

    return roi_img

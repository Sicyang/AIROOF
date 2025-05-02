🪜 Step 1: Install Required Packages

Before running the main detection and processing scripts, you need to install all required Python dependencies.

✅ File: requirements.txt

▶️How to install:

pip install -r requirements.txt

🪜 Step 2: Classify Images by Camera

Before starting the main detection and analysis tasks, you need to organize your raw images into subfolders based on their camera source. This step ensures that later processing can automatically find the correct camera configuration for each image.

✅ Script: sorted_camera.py

Example:

Original: datasets/Camera3_img123.jpg

After running: sorted_by_camera/Camera3/Camera3_img123.jpg

🪜 Step 3: Split Dataset into Train/Test/Val

After classifying the images by camera, you need to split them into training, testing, and validation sets for model training. This script ensures that each camera’s images are split individually, so every camera is represented in each subset.

✅ Script: movepictures.py

Default split ratio: Train 70% Test 20% Val 10%

You can adjust the ratios in the script parameters if needed.

▶️ How to Run

1️⃣ Make sure the source folder (sorted_by_camera/) is prepared from Step 1.

2️⃣ Default paths (you can change them in the script):

 source_folder = os.path.join(base_dir, "..", "sorted_by_camera")
 destination_root = os.path.join(base_dir, "datasets", "images")

3️⃣ Run the script:
python split_by_camera_individually.py

The console will print how many images were split into each set for each camera

🪜 Step 4: Preprocess Images

Before training the detection model, you need to preprocess all images to standardize their size, crop the ROI (region of interest), and enhance contrast. This script applies preprocessing to all images in the train, test, and val folders.

✅ Script: OpenCV.py

What it does:

Reads the camera configuration from a JSON file.

For each image:

Extracts the corresponding camera name from the filename.

Looks up the camera’s configuration (ROI points, reference image size, etc.).

Resizes the image to match the reference size.

Crops the ROI mask and applies histogram equalization to improve contrast in the ROI.

Fixes any invalid PNG format (if needed).

Overwrites the original images with the processed versions.

✅Folder structure:

This script processes images inside:

datasets/images/
├── train/
├── test/
├── val/

Each subset will be fully processed.

▶️ How to Run

1️⃣ Make sure your camera configuration JSON file is ready.

2️⃣ The script has these default paths (can be changed in the code):

config_path = "camera_config.json"
base_path = "datasets/images"

3️⃣ Run the script

🪜 Step 5: Generate Camera Configuration

To process images properly, each camera must have a configuration that defines its ROI (Region of Interest), reference image, and model paths. This script helps you interactively select the ROI and automatically generates or updates the camera_config.json file.

✅ Script: camera_config.py

What it does:

Scans a folder of standard images (typically one representative image per camera).

For each camera:

Detects the camera name from the filename.

Skips if already configured.

Uses OpenCV’s interactive window to let you draw the ROI on the image.

Saves the following into camera_config.json:

roi_points: The polygon of the selected ROI.

reference_image: The path to the reference image.

model_path_detect: Default YOLO detection model path.

model_path_segment: Default YOLO segmentation model path.

confidence_threshold and area_ratio_threshold: Default detection parameters.

Example config entry:

{
  "Camera3": {
    "model_path_detect": "runs/detect/train/weights/best.pt",
    "model_path_segment": "runs/segment/train/weights/best.pt",
    "reference_image": "datasets/standard/Camera3_img001.jpg",
    "roi_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "confidence_threshold": 0.2,
    "area_ratio_threshold": 0.1
  }
}

▶️ How to Run

1️⃣ Prepare a folder with one standard image per camera (e.g., datasets/standard/).

2️⃣ In the script, the default paths are:

image_folder = "datasets/standard"
config_save_path = "camera_config.json"

3️⃣ Run the script:

python generate_camera_config.py

✅ For each image:

A window will open showing the image.

Use your mouse to select the ROI (drag to draw a rectangle).

After selection, the ROI coordinates will be displayed in the console and saved automatically.

If the camera is already configured, it will be skipped.

🪜 Step 6: Annotate Images with Labelme

To train your detection or segmentation models, you need to annotate the images with proper labels. This step uses Labelme, a popular graphical image annotation tool.

Default folders:

Input images	./datasets/images

Output annotations	./datasets/labels

Your dataset is split into train, test, and val inside datasets/images.

You should save the corresponding .json annotation files in the same subfolder structure inside datasets/labels.

Example:

datasets/
├── images/
│   ├── train/
│   │   ├── Camera3_img001.jpg
│   ├── test/
│   └── val/
├── labels/
│   ├── train/
│   │   ├── Camera3_img001.json
│   ├── test/
│   └── val/

▶️ How to Annotate

1️⃣ Install Labelme if you haven’t already:

pip install labelme

2️⃣ Launch Labelme:

labelme

3️⃣ Open images from the datasets/images folder.

4️⃣ After annotating an image:

Save the .json file to the corresponding folder under datasets/labels.

Make sure the filename matches the image name (e.g., Camera3_img001.json for Camera3_img001.jpg).

🪜 Step 7: Train the YOLO Model

Once your images and labels are prepared, you can train the YOLOv8 model using the Ultralytics framework.

✅ Script: train.py

What it does:

Initializes a YOLO model with:

A YAML configuration file (e.g., yolov8.yaml).

Pre-trained weights (e.g., yolov8n.pt).

Trains the model using your labeled dataset for:

500 epochs (default in script).

Image size 640×640.

Default paths in the script:

model = YOLO("models/yolov8.yaml")

model = YOLO("models/yolov8n.pt")

model.train(data='models/data.yaml', epochs=500, imgsz=640)

▶️ How to Run

1️⃣ Make sure your data.yaml is properly set up (points to your images and labels).

2️⃣ Run the script:

python train.py

✅ Training progress (metrics, loss, mAP, etc.) will be printed in the console, and output folders (e.g., runs/detect/train/) will be created to store:

Best model weights (best.pt)

🪜 Step 8: Train the YOLO Segmentation Model

In addition to detection, you can train a segmentation model to detect and segment specific regions (e.g., leaf areas) in your images.

✅ Script: segment.py

What it does:

Initializes a YOLO segmentation model with:

A YAML configuration file (e.g., yolov8n-seg.yaml).

Pre-trained weights (e.g., yolov8n-seg.pt).

Trains the segmentation model using your labeled dataset for:

500 epochs (default in script).

Image size 640×640.

Default paths in the script:

model = YOLO("models/yolov8n-seg.yaml")

model = YOLO("models/yolov8n-seg.pt")

model.train(data='models/coco128-seg.yaml', epochs=500, imgsz=640)

▶️ How to Run

1️⃣ Make sure your segmentation labels are prepared (typically in COCO JSON format or YOLO segmentation format).

2️⃣ Confirm your coco128-seg.yaml (or your own YAML file) is set up properly, e.g.:

path: ../datasets
train: images/train
val: images/val
test: images/test
names:
  0: leaf

3️⃣ Run the script:

python train_yolo_seg.py

✅ Output will be saved under runs/segment/train/, including:

Best model weights (best.pt)

Training logs and validation results

🪜 Step 9: Run Detection, Segmentation, and Generate Report

This script performs automated detection and segmentation on a batch of images, analyzes the results, and generates a comprehensive PDF report summarizing the findings for each image.

✅ Script: predict.py

What it does:

For each image in the running test folder:

1️⃣ Runs detection using the trained YOLO detection model.

2️⃣ If leaves are detected (above confidence threshold), runs the segmentation model to analyze leaf coverage.

3️⃣ Calculates:
- Leaf area ratio within the ROI.
- Whether leaves overlap with critical drainage areas.

4️⃣ Determines the warning level:
- Green: No cleaning needed.
- Yellow: Leaf accumulation detected (either significant or near drainage).
- Red: Heavy leaf accumulation; immediate cleaning required.

5️⃣ Generates:
- A PDF report (includes timestamp, image, results text).
- A TXT summary file with numeric metrics.

6️⃣ Merges all individual PDFs + legend into a single report.

Default folders:

Purpose	Path
Input images	datasets/running test/
Camera config JSON	camera_config.json
Results (PDF + TXT)	results/
YOLO detect runs	runs/detect/
YOLO segment runs	runs/segment/

Example output:

results/
├── legend.pdf
├── result_20240502_123456.pdf
├── result_20240502_123500.pdf
├── merged_results.pdf
├── results_20240502_123456.txt
├── results_20240502_123500.txt

▶️ How to Run

1️⃣ Make sure:

Your camera configuration JSON is ready.

Your trained detection + segmentation models are in place.

Images to analyze are in datasets/running test/.

2️⃣ Run the script:

python predict.py

✅ As it runs, you will see logs for:

Preprocessing images.

Running detection + segmentation.

Writing PDFs and TXT summaries.

At the end, all PDFs will be merged into a single report (e.g., merged_results.pdf).
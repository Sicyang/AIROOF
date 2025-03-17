import json
import os
from tqdm import tqdm

def convert_label_json(json_dir, save_dir, classes):
    """
    Convert JSON label format to YOLO TXT format with normalized coordinates.

    Args:
        json_dir (str): Path to the directory containing JSON annotation files.
        save_dir (str): Path to save the converted TXT files.
        classes (str or list): A comma-separated string or list of class names.

    Example usage:
        convert_label_json("datasets/labels/jsons", "datasets/labels/txts", "cat,dog")
    """
    if isinstance(classes, str):
        classes = classes.split(',')

    os.makedirs(save_dir, exist_ok=True)
    json_paths = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    for json_path in tqdm(json_paths, desc="Converting JSON to TXT"):
        json_file = os.path.join(json_dir, json_path)
        with open(json_file, 'r') as f:
            json_dict = json.load(f)

        h, w = json_dict['imageHeight'], json_dict['imageWidth']
        txt_path = os.path.join(save_dir, json_path.replace('.json', '.txt'))

        with open(txt_path, 'w') as txt_file:
            for shape_dict in json_dict.get('shapes', []):
                label = shape_dict['label']
                if label not in classes:
                    continue  # Skip labels that are not in the defined class list

                label_index = classes.index(label)
                points = shape_dict['points']

                # Normalize points
                points_nor_list = [str(point[0] / w) + " " + str(point[1] / h) for point in points]
                points_nor_str = ' '.join(points_nor_list)

                label_str = f"{label_index} {points_nor_str}\n"
                txt_file.writelines(label_str)

        print(f"Converted: {json_file} -> {txt_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert JSON annotations to YOLO TXT format.')
    parser.add_argument('--json-dir', type=str, required=True, help='Path to JSON annotations directory.')
    parser.add_argument('--save-dir', type=str, required=True, help='Path to save converted TXT files.')
    parser.add_argument('--classes', type=str, required=True, help='Comma-separated list of class names.')

    args = parser.parse_args()
    convert_label_json(args.json_dir, args.save_dir, args.classes)


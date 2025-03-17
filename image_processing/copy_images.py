import os
import shutil

def copy_camera_images(source_base, destination):
    """
    Copies images containing 'Camera3' from source_base to destination.
    """
    os.makedirs(destination, exist_ok=True)

    for i in range(1, 7):  # Loop through RP-2024-1 to RP-2024-6
        source_folder = os.path.join(source_base, f"RP-2024-{i}")
        if os.path.exists(source_folder):
            for file_name in os.listdir(source_folder):
                if "Camera3" in file_name and file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    source_path = os.path.join(source_folder, file_name)
                    destination_path = os.path.join(destination, file_name)
                    shutil.copy2(source_path, destination_path)
                    print(f"Copied: {source_path} -> {destination_path}")

    print("All Camera images copied successfully.")

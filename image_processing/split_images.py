import os
import random
import shutil


def split_images(source_dir, test_dir, val_dir, test_ratio=0.2, val_ratio=0.2):
    """
    Splits images from source_dir into train, test, and validation sets.
    """
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_files = os.listdir(source_dir)
    random.shuffle(all_files)

    num_test = int(len(all_files) * test_ratio)
    num_val = int(len(all_files) * val_ratio)

    test_files = all_files[:num_test]
    val_files = all_files[num_test:num_test + num_val]
    train_files = all_files[num_test + num_val:]

    for file_name in test_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(test_dir, file_name))

    for file_name in val_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(val_dir, file_name))

    print(
        f"File distribution complete! Test set: {len(test_files)} images, Validation set: {len(val_files)} images, Training set: {len(train_files)} images.")

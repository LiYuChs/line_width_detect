import cv2
import os


def imwrite_check(path, img):
    """Helper function to save an image and check if it was successful."""
    check_path_exists(path)
    success = cv2.imwrite(str(path), img)
    if not success:
        print(f"Failed to save image at {path}")
    return success


def check_path_exists(folder_path):
    if os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    else:
        dir_path = os.path.dirname(folder_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

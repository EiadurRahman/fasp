# utils.py
import cv2
import os

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}. Check if it's a valid image file.")
    return img

def save_image(path, img):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success = cv2.imwrite(path, img)
    if not success:
        raise ValueError(f"Failed to save image to {path}")
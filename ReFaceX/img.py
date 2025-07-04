# img.py

from pathlib import Path
import utils
from swap import FaceSwapper

SUPPORTED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

def is_image_file(path):
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXT

def process_image_file(image_path, source_face, swapper, output_dir):
    """
    Process a single image: perform face swap and save the result.
    """
    try:
        img = utils.load_image(image_path)
        swapped = swapper.swap_all_faces(img, source_face)
        filename = Path(image_path).stem
        output_path = Path(output_dir) / f"swapped_{filename}.jpg"
        utils.save_image(str(output_path), swapped)
        print(f"[✓] Saved swapped image: {output_path}")
    except Exception as e:
        print(f"❌ Failed to process image {image_path}: {e}")

def process_images_in_directory(input_dir, source_face, swapper, output_dir):
    """
    Process all supported images in the input directory.
    """
    image_files = [f for f in Path(input_dir).iterdir() if is_image_file(f)]
    
    if not image_files:
        print(f"⚠️ No supported image files found in {input_dir}")
        return

    print(f"🖼️ Found {len(image_files)} image(s) to process in {input_dir}")

    for image_file in image_files:
        process_image_file(str(image_file), source_face, swapper, output_dir)

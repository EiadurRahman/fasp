# main.py
import os
import time
import argparse
from face_swapper import FaceSwapper
from utils import load_image, save_image
from tqdm import tqdm
from pathlib import Path

# Enable verbose OpenVINO logs (optional)
os.environ["OPENVINO_VERBOSE"] = "1"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def is_image_file(path):
    return path.suffix.lower() in IMAGE_EXTS

def get_target_images(target_path):
    path = Path(target_path)
    if path.is_file() and is_image_file(path):
        return [path]
    elif path.is_dir():
        return sorted([p for p in path.glob("*") if is_image_file(p)])
    else:
        raise ValueError(f"Invalid target path: {target_path}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Face swap using InsightFace + ONNX")
    parser.add_argument("--source", required=True, help="Path to source image")
    parser.add_argument("--target", required=True, help="Path to target image or directory")
    parser.add_argument("--output", default="assets/output", help="Output directory path")
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    source_img = load_image(str(source_path))
    swapper = FaceSwapper()

    target_images = get_target_images(target_path)
    if not target_images:
        print("No valid target images found.")
        return

    print(f"✓ Found {len(target_images)} image(s) to process.")
    print("⚙️  Starting face swap...")

    # Time first image to estimate
    init_time = time.time()
    test_target_img = load_image(str(target_images[0]))
    _ = swapper.swap(test_target_img, source_img)
    est_time = time.time() - init_time
    total_est = est_time * len(target_images)
    print(f"⏱ Estimated total time: {total_est:.2f} seconds ({total_est/60:.1f} min)")

    for img_path in tqdm(target_images, desc="🔄 Swapping faces", unit="img"):
        try:
            target_img = load_image(str(img_path))
            result = swapper.swap(target_img, source_img)
            output_name = f"s_{source_path.stem}-{img_path.stem}{img_path.suffix}"
            output_path = output_dir / output_name
            save_image(str(output_path), result)
        except Exception as e:
            print(f"⚠️ Failed to process {img_path.name}: {e}")

    print("✅ Face swapping completed.")

if __name__ == "__main__":
    main()

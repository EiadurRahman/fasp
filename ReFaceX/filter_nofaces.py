import os
import shutil
from utils import face_detected  # make sure utils.py is in the same dir or PYTHONPATH

def move_noface_images(image_dir, providers=['CPUExecutionProvider']):
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"❌ Not a valid directory: {image_dir}")

    noface_dir = os.path.join(image_dir, "nofaces")
    os.makedirs(noface_dir, exist_ok=True)

    total = 0
    moved = 0

    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        total += 1
        try:
            if not face_detected(file_path, providers=providers):
                print(f"🚫 No face: {filename}")
                shutil.move(file_path, os.path.join(noface_dir, filename))
                moved += 1
            else:
                print(f"✅ Face found: {filename}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    print(f"\n📦 Processed {total} images. Moved {moved} to 'nofaces/'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python filter_nofaces.py /path/to/image_dir")
        exit(1)

    image_dir = sys.argv[1]
    move_noface_images(image_dir)

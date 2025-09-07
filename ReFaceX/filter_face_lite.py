import cv2
import os
import shutil
import sys

PROTO_TXT = "models/deploy.prototxt"
MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

def download_model():
    import urllib.request
    if not os.path.exists(PROTO_TXT):
        print("Downloading deploy.prototxt...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            PROTO_TXT)
    if not os.path.exists(MODEL):
        print("Downloading res10_300x300_ssd_iter_140000.caffemodel...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            MODEL)

def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)
    return net

def detect_face(net, image_path, conf_threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return False
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            return True
    return False

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

def main(input_dir, invert=False):
    if not os.path.exists(input_dir):
        print("Input directory does not exist.")
        return

    download_model()
    net = load_model()

    target_dir = "noface" if invert else "face"
    target_path = os.path.join(input_dir, target_dir)
    os.makedirs(target_path, exist_ok=True)

    copied_files = []

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            ext = filename.lower().split('.')[-1]
            if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
                face_found = detect_face(net, filepath)
                # Copy based on invert flag
                if (face_found and not invert) or (not face_found and invert):
                    dest_path = os.path.join(target_path, filename)
                    shutil.copy2(filepath, dest_path)
                    copied_files.append(filename)
                    action = "Face detected" if face_found else "No face detected"
                    print(f"{action}: {filename} -> copied to {target_dir}/")

    sorted_files = quicksort(copied_files)
    print(f"\nSorted files in {target_dir}/:")
    for f in sorted_files:
        print(f)

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: python face_filter.py /path/to/images [--inv]")
        sys.exit(1)
    input_dir = sys.argv[1]
    invert = False
    if len(sys.argv) == 3:
        invert = sys.argv[2] == "--inv"
    main(input_dir, invert)

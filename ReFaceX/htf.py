import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
import os

# ------------------------------------------
# 🚀 Step 1: Initialize face detector and swapper model
# ------------------------------------------

# Load the ONNX face swapping model (inswapper_128)
# This model contains logic to blend source face identity into target face
swapper = model_zoo.get_model("models/inswapper_128.onnx", providers=["CPUExecutionProvider"])

# Load the face detection and recognition model
# This detects faces and gives embeddings (512-dimensional vectors that encode identity)
detector = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=0)  # ctx_id=0 for CPU or GPU, -1 for CPU-only

# ------------------------------------------
# 📸 Step 2: Load source and target images
# ------------------------------------------

# Load source image (this is the face we want to transplant)
source_img_path = "/home/eiadurrahman/Desktop/faceswap/files/input/source/src_1.jpg"
source_img = cv2.imread(source_img_path)

if source_img is None:
    raise FileNotFoundError(f"❌ Source image not found: {source_img_path}")

# Load target image (the face(s) we want to replace)
target_img_path = "/home/eiadurrahman/Desktop/faceswap/files/msmoon/target.JPG"
target_img = cv2.imread(target_img_path)

if target_img is None:
    raise FileNotFoundError(f"❌ Target image not found: {target_img_path}")

# At this point:
# - source_img and target_img are NumPy arrays with dtype=uint8 and shape=(H, W, 3)
# - Colors are in BGR format (because OpenCV uses BGR)

# ------------------------------------------
# 🧠 Step 3: Extract the source face identity
# ------------------------------------------

# FaceAnalysis.get() returns a list of detected faces with:
# - bounding box
# - facial landmarks
# - 512D embedding (`embedding`)
# - normalized embedding (`normed_embedding`)
source_faces = detector.get(source_img)

if not source_faces:
    raise ValueError("❌ No face detected in source image.")

# Use the first detected face (assume it’s the most prominent one)
source_face = source_faces[0]

# source_face.normed_embedding is the compressed identity vector (unit-normalized)

# ------------------------------------------
# 🕵️ Step 4: Detect faces in the target image
# ------------------------------------------

target_faces = detector.get(target_img)

if not target_faces:
    raise ValueError("❌ No face detected in target image.")

# ------------------------------------------
# 🤖 Step 5: Perform face swap on each target face
# ------------------------------------------

# Make a copy so original is untouched
result_img = target_img.copy()

for i, target_face in enumerate(target_faces):
    try:
        # swapper.get() modifies result_img by injecting source identity into target face
        result_img = swapper.get(
            result_img,
            source_face=source_face,
            target_face=target_face
        )
        print(f"✅ Swapped face #{i + 1}")
    except Exception as e:
        print(f"⚠️ Failed to swap face #{i + 1}: {e}")

# ------------------------------------------
# 💾 Step 6: Save output
# ------------------------------------------

output_path = "trash/swapped.jpg"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

success = cv2.imwrite(output_path, result_img)
if success:
    print(f"✅ Saved final swapped image to: {output_path}")
else:
    print("❌ Failed to save output image")

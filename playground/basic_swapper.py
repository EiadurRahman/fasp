import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from types import SimpleNamespace

# Load target image
target_img = cv2.imread("assets/input/target/trgt_1.jpg")
assert target_img is not None, "❌ Failed to load target image"

# Load super face embedding
super_face = np.load("ignr/super_face.npz")["super_face"]

# Initialize InsightFace Face Detector
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0)

# Detect faces in target image
faces = face_analyzer.get(target_img)
assert len(faces) > 0, "❌ No faces found in target image"

# Load face swapper model
swapper = model_zoo.get_model("models/inswapper_128.onnx", providers=["CPUExecutionProvider"])

# Apply face swap using super_face embedding
face = faces[0]  # Just the first face

# Create a fake source face object with only the embedding
source_face = SimpleNamespace(normed_embedding=super_face)

# Now pass it to swapper
swapped_img = swapper.get(target_img, face, source_face)

# Save result
cv2.imwrite("swapped.jpg", swapped_img)
print("✅ Face swapped image saved as swapped.jpg")

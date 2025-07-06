from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_embedding(image_np):
    faces = app.get(image_np)
    if faces:
        return faces[0].embedding
    return None

def face_similarity(img1_np, img2_np):
    emb1 = get_embedding(img1_np)
    emb2 = get_embedding(img2_np)

    if emb1 is None or emb2 is None:
        return 0.0  # No face detected in one of the images

    # cosine similarity gives value in [-1, 1], we map it to [0, 1]
    sim = cosine_similarity([emb1], [emb2])[0][0]
    return (sim + 1) / 2

# Usage
img1 = cv2.imread('/home/eiadurrahman/faceswap/files/input/source/shreya_2.jpg')
img2 = cv2.imread('/home/eiadurrahman/faceswap/files/input/target/aish.jpg')

score = face_similarity(img1, img2)
print(f"Similarity score: {score:.2f}")

import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
from utils import face_detected, save_image
from filter_face_lite import download_model, load_model, detect_face
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
import numpy as np


def extract_video_frames(video_path, frame_dir, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    os.makedirs(frame_dir, exist_ok=True)
    frame_id = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_skip == 0:
            filename = os.path.join(frame_dir, f"frame_{frame_id:06d}.jpg")
            save_image(filename, frame)
            saved += 1
        frame_id += 1
    cap.release()
    return saved


def detect_faces_in_frames(frame_dir, faces_dir):
    download_model()
    net = load_model()

    os.makedirs(faces_dir, exist_ok=True)
    for frame in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, frame)
        if detect_face(net, frame_path):
            shutil.copy2(frame_path, os.path.join(faces_dir, frame))


def sort_faces_by_identity(faces_dir):
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    embeddings = []
    paths = []

    for file in os.listdir(faces_dir):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(faces_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if faces:
                embeddings.append(faces[0].embedding)
                paths.append(img_path)

    if not embeddings:
        print("❌ No face embeddings found to cluster")
        return

    X = np.array(embeddings)
    db = DBSCAN(eps=0.5, min_samples=3, metric='cosine').fit(X)
    labels = db.labels_
    print(f"🔍 Found {len(set(labels)) - (1 if -1 in labels else 0)} unique identities")

    for label, path in zip(labels, paths):
        if label == -1:
            continue
        label_dir = os.path.join(faces_dir, f"face_{label}")
        os.makedirs(label_dir, exist_ok=True)
        shutil.move(path, os.path.join(label_dir, os.path.basename(path)))


def extract_and_sort_faces(video_path):
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    base_dir = video_path.parent
    temp_dir = base_dir / "temp"
    faces_dir = base_dir / "faces_src"

    print("📦 Extracting frames...")
    extract_video_frames(str(video_path), str(temp_dir))

    print("🔎 Detecting faces in frames...")
    detect_faces_in_frames(str(temp_dir), str(faces_dir))

    print("🧬 Sorting faces by person identity...")
    sort_faces_by_identity(str(faces_dir))

    print("✅ Done. Faces sorted into:", faces_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    args = parser.parse_args()

    extract_and_sort_faces(args.video)

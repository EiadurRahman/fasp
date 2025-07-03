import zipfile
import os
import shutil
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

FSZ_TEMP_DIR = 'tem'

def extract_fsz(fsz_path, extract_dir=FSZ_TEMP_DIR):
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(fsz_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    images = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return images

def get_face_embeddings(images, app):
    embeddings = []

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"⚠️ No face found in {img_path}")
            continue

        # Just take the first detected face
        emb = faces[0].normed_embedding
        if emb is not None:
            embeddings.append(emb)

    return embeddings

def average_embeddings(embeddings):
    if not embeddings:
        raise ValueError("❌ No valid embeddings found.")
    
    stacked = np.vstack(embeddings)
    avg = np.mean(stacked, axis=0)
    norm = avg / np.linalg.norm(avg)
    return norm

def save_embedding(embedding, output_path):
    np.savez_compressed(output_path, super_face=embedding)
    print(f"✅ Super face embedding saved to {output_path}")

def main(fsz_path, output_path='super_face.npz'):
    print(f"📦 Processing: {fsz_path}")
    images = extract_fsz(fsz_path)
    
    print(f"🔍 Found {len(images)} image(s) inside the FaceSet.")

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    embeddings = get_face_embeddings(images, app)
    print(f"🧠 Got {len(embeddings)} valid face embedding(s).")

    if len(embeddings) == 0:
        print("❌ No embeddings found. Exiting.")
        return

    super_face = average_embeddings(embeddings)
    save_embedding(super_face, output_path)

    # Cleanup
    shutil.rmtree(FSZ_TEMP_DIR)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create super face embedding from .fsz file")
    parser.add_argument("fsz_path", help="Path to .fsz file (ZIP of face images)")
    parser.add_argument("--output", default="super_face.npz", help="Output file for super face embedding")
    args = parser.parse_args()

    main(args.fsz_path, args.output)

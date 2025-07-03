# utils.py
import os
import cv2
import zipfile
import shutil
import numpy as np
from types import SimpleNamespace
from insightface.app import FaceAnalysis

FSZ_TEMP_DIR = "temp_fsz_extract"

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}. Check if it's a valid image file.")
    return img

def clear_temp():
    if os.path.exists(FSZ_TEMP_DIR):
        try:
            print(f"🧹 Clearing temporary directory: {FSZ_TEMP_DIR}")
            shutil.rmtree(FSZ_TEMP_DIR)
        except Exception as e:
            print(f"❌ Failed to remove temp directory: {e}")
    else:
        print("✅ No temp directory found.")


def save_image(path, img):
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success = cv2.imwrite(path, img)
    if not success:
        raise ValueError(f"Failed to save image to {path}")

def load_super_face(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Super face embedding file not found: {path}")
    
    data = np.load(path)
    if "super_face" not in data:
        raise KeyError("File does not contain 'super_face' key.")
    
    emb = data["super_face"]
    return SimpleNamespace(normed_embedding=emb)

def extract_fsz_images(fsz_path):
    if os.path.exists(FSZ_TEMP_DIR):
        shutil.rmtree(FSZ_TEMP_DIR)
    os.makedirs(FSZ_TEMP_DIR, exist_ok=True)

    with zipfile.ZipFile(fsz_path, 'r') as zip_ref:
        zip_ref.extractall(FSZ_TEMP_DIR)

    image_paths = [
        os.path.join(FSZ_TEMP_DIR, f)
        for f in os.listdir(FSZ_TEMP_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return image_paths

def get_embedding_from_image(img, app: FaceAnalysis):
    faces = app.get(img)
    if not faces:
        raise ValueError("❌ No face detected in image.")

    return faces[0].normed_embedding

def process_source(source_path):
    """
    Process any type of source input (image, .fsz, .npz) and return a valid SimpleNamespace(normed_embedding=...)
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    ext = os.path.splitext(source_path)[1].lower()

    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    # CASE 1: super_face.npz
    if ext == ".npz":
        return load_super_face(source_path)

    # CASE 2: faceset.fsz (ZIP of face images)
    elif ext == ".fsz" or ext == ".zip":
        image_paths = extract_fsz_images(source_path)
        embeddings = []

        for path in image_paths:
            try:
                img = load_image(path)
                emb = get_embedding_from_image(img, app)
                embeddings.append(emb)
            except Exception as e:
                print(f"⚠️ Skipping {path}: {e}")

        if not embeddings:
            raise ValueError("❌ No valid face embeddings found in FaceSet.")

        avg_emb = np.mean(np.vstack(embeddings), axis=0)
        norm_emb = avg_emb / np.linalg.norm(avg_emb)

        return SimpleNamespace(normed_embedding=norm_emb)

    # CASE 3: single image
    elif ext in [".jpg", ".jpeg", ".png"]:
        img = load_image(source_path)
        emb = get_embedding_from_image(img, app)
        return SimpleNamespace(normed_embedding=emb)

    else:
        raise ValueError(f"Unsupported source file format: {ext}")
    
    




if __name__ == "__main__":
    try:
        source = "/home/eiadurrahman/Desktop/faceswap/ignr/faceset/nehal.fsz"  # Change to your test file
        embedding = process_source(source).encode('utf-8')
        print("Embedding processed successfully")
        with open("embedding.ReFX", "wb") as f:
            f.write(embedding)
    

    except Exception as e:
        print("Error processing source:", e)
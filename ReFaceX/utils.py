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

def face_detected(source, min_face_size=60, max_yaw=30, providers=['CPUExecutionProvider']):
    """
    Detect if there's at least one *valid* face in the image.
    
    Parameters:
    - source: image path or numpy array
    - min_face_size: minimum width or height of face bbox to accept
    - max_yaw: optional, max absolute yaw angle to consider a face valid
    - providers: ONNX runtime providers

    Returns:
    - True if a good face is detected, False otherwise
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        img = load_image(source)
    elif isinstance(source, np.ndarray):
        img = source
    else:
        raise TypeError("Input must be a file path or an image array.")

    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0)

    faces = app.get(img)
    if not faces:
        return False

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        w = x2 - x1
        h = y2 - y1

        # Reject small faces
        if w < min_face_size or h < min_face_size:
            continue

        # Optional: filter by yaw (rotation left/right)
        yaw = abs(face.pose[0]) if hasattr(face, 'pose') else 0
        if yaw > max_yaw:
            continue

        return True  # Found a usable face

    return False  # No good face found



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

def create_npz(source_path, output_path="out.npz",providers=['CPUExecutionProvider']):
    """
    Converts a .fsz (zip of face images) or a directory of images to an .npz file containing a 'super_face' embedding.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"❌ Source path not found: {source_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        

    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0)

    # Handle ZIP (.fsz)
    if source_path.lower().endswith((".fsz", ".zip")):
        print(f"📦 Extracting and processing images from zip: {source_path}")
        image_paths = extract_fsz_images(source_path)
    
    # Handle image directory
    elif os.path.isdir(source_path):
        print(f"📂 Processing image directory: {source_path}")
        image_paths = [
            os.path.join(source_path, f)
            for f in os.listdir(source_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    
    else:
        raise ValueError(f"❌ Unsupported source format: {source_path}")

    if not image_paths:
        raise ValueError("❌ No valid images found.")

    embeddings = []
    for path in image_paths:
        try:
            img = load_image(path)
            emb = get_embedding_from_image(img, app)
            embeddings.append(emb)
        except Exception as e:
            print(f"⚠️ Skipping {path}: {e}")

    if not embeddings:
        raise ValueError("❌ No usable face embeddings were extracted.")

    avg_emb = np.mean(np.vstack(embeddings), axis=0)
    norm_emb = avg_emb / np.linalg.norm(avg_emb)

    np.savez_compressed(output_path, super_face=norm_emb.astype(np.float32))
    print(f"✅ Saved averaged embedding to: {output_path}")



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

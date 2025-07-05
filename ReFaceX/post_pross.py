import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from pathlib import Path

# --- Load models ---

detector = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
detector.prepare(ctx_id=0)

# Load ONNX ESRGAN session
esrgan_session = ort.InferenceSession("models/realesrgan-x4.onnx", providers=['CPUExecutionProvider'])

def preprocess_esrgan(img: np.ndarray) -> np.ndarray:
    """
    Prepares a cropped face image for ESRGAN ONNX input.
    ESRGAN expects input shape: [1, 3, H, W] with float32 and normalized [0,1].
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return img

def postprocess_esrgan(output: np.ndarray) -> np.ndarray:
    """
    Converts ESRGAN ONNX output back to BGR uint8 image.
    Output shape: [1, 3, H_up, W_up]
    """
    output = output.squeeze(0)   # remove batch dim
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

def upscale_face_onnx_esrgan(face_img: np.ndarray) -> np.ndarray:
    """
    Runs the ESRGAN ONNX model to upscale a face image.
    """
    input_tensor = preprocess_esrgan(face_img)
    outputs = esrgan_session.run(None, {esrgan_session.get_inputs()[0].name: input_tensor})
    upscaled_img = postprocess_esrgan(outputs[0])
    return upscaled_img

def main(input_path: str, output_path: str):
    # Load image BGR uint8
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image at {input_path}")

    # Detect faces
    faces = detector.get(img)
    if not faces:
        raise RuntimeError("No faces detected in the image")

    # Crop first face by bbox
    bbox = faces[0].bbox.astype(int)
    x1, y1, x2, y2 = bbox
    face_crop = img[y1:y2, x1:x2]

    # Upscale face crop using ONNX ESRGAN
    upscaled_face = upscale_face_onnx_esrgan(face_crop)

    # Save upscaled face
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, upscaled_face)

    print(f"Upscaled face saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python upscale_face.py <input_image> <output_image>")
        exit(1)

    input_img_path = sys.argv[1]
    output_img_path = sys.argv[2]

    main(input_img_path, output_img_path)

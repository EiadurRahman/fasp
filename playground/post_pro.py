import cv2
import os
import onnxruntime as ort
import numpy as np
import time

def preprocess_image(image_path, size=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if size:
        img = cv2.resize(img, (size, size))

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_output(output):
    img = output.squeeze(0)
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def run_onnx_model(image_path, model_path, size=None):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    input_data = preprocess_image(image_path, size)
    output = session.run(None, {input_name: input_data})[0]
    output_img = postprocess_output(output)
    return output_img

def save_output_image(input_path, output_image):
    dirname, basename = os.path.split(input_path)
    name, ext = os.path.splitext(basename)
    output_path = os.path.join(dirname, f"pro_{name}{ext}")
    cv2.imwrite(output_path, output_image)
    print(f"Saved processed image at: {output_path}")

def main():
    # === 🔧 Hardcoded paths for testing ===
    input_path = "assets/input/source/59.png"
    model_path = "models/GFPGANv1.4.onnx"  # this one works the great but image has to be square.
    # model_path = "models/codeformer.onnx" # this doesn't work
    # model_path = "models/Real-ESRGAN-x4plus.onnx"  # needs 128 | prettry bad results
    # model_path = "models/RestoreFormer.onnx"  # pretty bad results
    resize_to = 512 # Set to None if your model accepts variable input sizes

    init_time = time.time()
    output_img = run_onnx_model(input_path, model_path, resize_to)
    save_output_image(input_path, output_img)
    elapsed_time = time.time() - init_time
    print(f"Processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()

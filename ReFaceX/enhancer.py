# enhancer.py

import onnxruntime as ort
import numpy as np
import cv2
from insightface.app import FaceAnalysis

class Enhancer:
    def __init__(self, model_path='models/enhancer.onnx', provider='cpu', alpha=1.0):
        """
        model_path: path to the ONNX enhancement model (e.g., GPEN)
        provider: 'cpu' or 'cuda'
        alpha: blending factor between 0.0 (only original) and 1.0 (only enhanced)
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)

        if provider == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.face_app.prepare(ctx_id=0)

    def enhance_face_area(self, img):
        faces = self.face_app.get(img)
        if not faces:
            return img  # No face detected, return as-is

        result_img = img.copy()

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_crop = result_img[y1:y2, x1:x2]

            try:
                # Preprocess
                face_resized = cv2.resize(face_crop, (512, 512))
                face_input = (face_resized.astype(np.float32) / 127.5 - 1.0)[None].transpose(0, 3, 1, 2)

                # Run ONNX
                out = self.session.run(None, {self.input_name: face_input})[0]
                out = (np.clip(out.squeeze().transpose(1, 2, 0), -1, 1) + 1.0) * 127.5
                enhanced = out.astype(np.uint8)
                enhanced = cv2.resize(enhanced, (x2 - x1, y2 - y1))

                # Blend with original based on alpha
                blended = cv2.addWeighted(enhanced, self.alpha, face_crop, 1.0 - self.alpha, 0)

                result_img[y1:y2, x1:x2] = blended
            except Exception as e:
                print(f"⚠️ Enhancement failed on face region: {e}")
                continue

        return result_img

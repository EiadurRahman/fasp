# swap.py
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from types import SimpleNamespace

# === Provider Mapping ===
def map_provider(provider):
    provider = provider.lower()

    if provider == "cpu":
        return ["CPUExecutionProvider"]
    elif provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "tensorrt":
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    elif provider == "openvino":
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    elif provider == "directml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        print(f"⚠️ Unknown provider '{provider}', falling back to CPU")
        return ["CPUExecutionProvider"]

# === Main FaceSwapper Class ===
class FaceSwapper:
    def __init__(self, model_path, provider='cpu', detection_model='buffalo_l', device_id=0, force_gpu=True):
        self.provider = provider
        self.device_id = device_id
        self.force_gpu = force_gpu
        self.model_path = model_path

        try:
            exec_providers = map_provider(self.provider)

            self.face_analyzer = FaceAnalysis(name=detection_model, providers=exec_providers)
            self.face_analyzer.prepare(ctx_id=device_id if force_gpu else -1, det_size=(640, 640))

            self.swapper = model_zoo.get_model(model_path, providers=exec_providers)

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize FaceSwapper: {e}")

    def detect_faces(self, img):
        return self.face_analyzer.get(img)

    def swap_one_face(self, target_img, source_face):
        """
        Swaps only the first detected face for now.
        `source_face` must be a SimpleNamespace(normed_embedding=...)
        """
        faces = self.detect_faces(target_img)

        if not faces:
            raise ValueError("❌ No face detected in target image.")

        target_face = faces[0]  # Only one face supported for now

        result_img = target_img.copy()
        result_img = self.swapper.get(result_img, source_face=source_face, target_face=target_face)

        return result_img

    # Future slot: swap_all_faces(), mask_occlusion(), apply_postprocessing()




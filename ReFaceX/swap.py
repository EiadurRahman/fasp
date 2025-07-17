from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from types import SimpleNamespace
import numpy as np
import cv2

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
        faces = self.detect_faces(target_img)
        if not faces:
            raise ValueError("❌ No face detected in target image.")

        result_img = target_img.copy()
        face = faces[0]
        result_img = self.swapper.get(result_img, source_face=source_face, target_face=face)

        return result_img

    def swap_all_faces(self, target_img, source_face, gender='m'):
        """
        Swaps all detected faces in the target image using a single source face embedding.
        Filters by gender if specified:
        - 'f' = female only
        - 'm' = male only
        - 'a' = all (default)
        """
        gender = gender.lower()
        if gender not in ('f', 'm', 'a'):
            raise ValueError("Invalid gender value. Use 'f', 'm', or 'a'.")

        faces = self.detect_faces(target_img)
        if not faces:
            raise ValueError("❌ No face detected in target image.")

        result_img = target_img.copy()

        for i, face in enumerate(faces):
            try:
                if gender == 'f' and face.gender != 0:
                    continue
                elif gender == 'm' and face.gender != 1:
                    continue

                result_img = self.swapper.get(result_img, source_face=source_face, target_face=face)
            except Exception as e:
                print(f"⚠️ Skipping face #{i}: {e}")
                continue

        return result_img

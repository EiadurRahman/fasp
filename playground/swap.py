import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo
from types import SimpleNamespace




# === SELECT YOUR PROVIDER ===
provider = "cpu"       # CPU only
# provider = "cuda"      # NVIDIA GPU (CUDA)
# provider = "openvino"  # Intel GPU/CPU (OpenVINO)
# provider = "tensorrt"  # NVIDIA GPU (TensorRT)
# provider = "directml"  # AMD GPU (DirectML, Windows-only)
# provider = "cuda"        # 👈 UNCOMMENT your choice here


# === ONNX MODEL SELECTION ===
# Choose the ONNX model you want to use for face swapping. Do not delete this section.

# Available models:
# onnx_model = "inswapper_128.onnx"  # Default model best on so far
# onnx_model = "reswapper_128.onnx"  
# onnx_model = "reswapper_256.onnx"  
# onnx_model = "reswapper-1019500.onnx"  
# onnx_model = "reswapper-429500.onnx"  
# onnx_model = "reswapper_256-1567500_originalInswapperClassCompatible.onnx"  // this one does not work with inswapper class



def map_provider(provider):
    provider = provider.lower()

    try:
        import torch
        if torch.cuda.is_available():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
    except:

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

class faceswap:
    def __init__(self,model_path,detection_model='buffalo_l',provider='',device_id=0,force_gpu=True):
        self.force_gpu = force_gpu
        self.model_path = model_path
        self.provider = provider

        try:
            # selecting providers eg cpu, cuda,
            provider = map_provider(self.provider)

            # faceanlyzer 
            self.face_analyzer = FaceAnalysis(name=detection_model)
            self.face_analyzer.prepare(ctx_id=device_id if self.force_gpu else -1, det_size=(640, 640))
            # settingup swapper
            self.swapper = model_zoo.get_model(self.model_path)


        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceSwapper: {str(e)}")
        
    def swap(self, source_face_matrix, target_matrix):
        source_face = SimpleNamespace(normed_embedding=source_face_matrix)
        target_face = target_matrix

        # swapping
        result_img = target_face.copy()
        result_img = self.swapper.get(result_img,source_face=source_face,target_face=target_face)
        return result_img
    
if __name__ == '__main__':
    model_path = "models/inswapper_128.onnx"
    face_swapper = faceswap(model_path)

    import utils

    source_image = utils.load_image("assets/input/source/59.png")
    target_image = utils.load_image("assets/imgs/aish3.jpg")

    img = face_swapper.swap(source_image,target_image)

    utils.save('fucking_image.jpeg',img)


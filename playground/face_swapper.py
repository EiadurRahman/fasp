# face_swapper.py

import cv2
import numpy as np
import os
import logging

# === SELECT YOUR PROVIDER ===
provider = "cpu"       # CPU only
# provider = "cuda"      # NVIDIA GPU (CUDA)
# provider = "openvino"  # Intel GPU/CPU (OpenVINO)
# provider = "tensorrt"  # NVIDIA GPU (TensorRT)
# provider = "directml"  # AMD GPU (DirectML, Windows-only)
# provider = "cuda"        # 👈 UNCOMMENT your choice here

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    from insightface.model_zoo.inswapper import INSwapper
    # from insightface.model_zoo.reswapper import Reswapper

except ImportError:
    raise ImportError("InsightFace library is required. Install with: pip install insightface")

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("ONNX Runtime is required. Install with: pip install onnxruntime")

# === ONNX MODEL SELECTION ===
# Choose the ONNX model you want to use for face swapping. Do not delete this section.

# Available models:
onnx_model = "inswapper_128.onnx"  # Default model best on so far
# onnx_model = "reswapper_128.onnx"  
# onnx_model = "reswapper_256.onnx"  
# onnx_model = "reswapper-1019500.onnx"  
# onnx_model = "reswapper-429500.onnx"  
# onnx_model = "reswapper_256-1567500_originalInswapperClassCompatible.onnx"  // this one does not work with inswapper class


# Map string to ONNX providers
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
    def __init__(self, model_name=onnx_model, detection_model="buffalo_l", device_id=0, force_gpu=True):
        self.force_gpu = force_gpu
        self.setup_logging()
        self.check_gpu_availability()

        try:
            providers = map_provider(provider)
            


            # If primary provider is not available, fallback
            if providers[0] not in ort.get_available_providers():
                print(f"❌ Requested provider '{providers[0]}' is not available. Falling back to CPU.")
                providers = ["CPUExecutionProvider"]

            self.face_analyzer = FaceAnalysis(name=detection_model)
            self.face_analyzer.prepare(ctx_id=device_id if self.force_gpu else -1, det_size=(640, 640))

            # self.swapper = get_model(model_name, download=True, download_zip=True)
            model_path = 'models/inswapper_128.onnx'
            self.swapper = get_model(model_path, download=True, download_zip=True)

            # ⚙️ Prevent double-initialization
            if hasattr(self.swapper, 'session') and self.swapper.session is not None:
                print("ℹ️ Swapper already initialized. Skipping duplicate prepare.")
            else:
                if self.force_gpu and 'OpenVINOExecutionProvider' in providers:
                    try:
                        provider_options = [{
                            'device_type': 'GPU_FP16',
                            'precision': 'FP16',
                            'enable_dynamic_shapes': True,
                            'num_streams': 1,
                            'cache_dir': './openvino_cache'
                        }]
                        self.swapper.prepare(
                            ctx_id=device_id,
                            providers=providers,
                            provider_options=provider_options
                        )
                        print("✓ Face swapper initialized with OpenVINO GPU provider")
                    except Exception as e:
                        print(f"⚠️ OpenVINO GPU failed, falling back to CPU: {e}")
                        providers = ["CPUExecutionProvider"]

                        # CPU fallback
                        session_options = ort.SessionOptions()
                        session_options.intra_op_num_threads = os.cpu_count()
                        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                        self.swapper.prepare(
                            ctx_id=-1,
                            providers=providers,
                            session_options=session_options
                        )
                        print("✓ Face swapper initialized with optimized CPU session")
                else:
                    # CPU or other non-OpenVINO provider
                    session_options = ort.SessionOptions()
                    session_options.intra_op_num_threads = os.cpu_count()
                    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                    self.swapper.prepare(
                        ctx_id=-1,
                        providers=providers,
                        session_options=session_options
                    )
                    print("✓ Face swapper initialized with optimized CPU session")


            self.log_active_providers()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceSwapper: {str(e)}")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def check_gpu_availability(self):
        available_providers = ort.get_available_providers()
        print("🔍 Available ONNX Runtime providers:")
        for p in available_providers:
            print(f"  - {p}")

        mapped_providers = map_provider(provider)
        if mapped_providers[0] not in available_providers:
            print(f"⚠️ '{mapped_providers[0]}' not available in ONNX Runtime. Some features may fallback to CPU.")


    def log_active_providers(self):
        try:
            if hasattr(self.swapper, 'session'):
                active = self.swapper.session.get_providers()
                print(f"🔄 Active providers in swapper: {active}")
            else:
                print("ℹ️ Cannot access session to log providers")
        except Exception as e:
            print(f"ℹ️ Failed to log active providers: {e}")

    def test_gpu_inference(self):
        try:
            test_input = np.random.rand(1, 3, 128, 128).astype(np.float32)
            test_session = ort.InferenceSession(
                self.swapper.model_file if hasattr(self.swapper, 'model_file') else None,
                providers=[provider, "CPUExecutionProvider"]
            )
            active = test_session.get_providers()[0]
            print(f"✅ Test session using: {active}")
            return active.lower() == provider.lower()
        except Exception as e:
            print(f"❌ GPU inference test failed: {e}")
            return False

    def swap(self, target_img, source_img):
        if target_img is None or source_img is None:
            raise ValueError("Both target and source images must be provided")
        if len(target_img.shape) != 3 or target_img.shape[2] != 3:
            raise ValueError("Target image must be a 3-channel BGR image")
        if len(source_img.shape) != 3 or source_img.shape[2] != 3:
            raise ValueError("Source image must be a 3-channel BGR image")

        try:
            src_faces = self.face_analyzer.get(source_img)
            dst_faces = self.face_analyzer.get(target_img)
        except Exception as e:
            raise RuntimeError(f"Face detection failed: {str(e)}")

        if not src_faces:
            raise ValueError("No faces detected in source image")
        if not dst_faces:
            raise ValueError("No faces detected in target image")

        source_face = src_faces[0]
        result_img = target_img.copy()

        for target_face in dst_faces:
            try:
                if "inswapper" in onnx_model.lower():
                    result_img = self.swapper.get(result_img, target_face, source_face )#, paste_back=True)
                else:
                    result_img = self.swapper.get(result_img, target_face, source_face)

            except Exception as e:
                print(f"⚠️ Failed to swap one face: {str(e)}")
                continue

        return result_img

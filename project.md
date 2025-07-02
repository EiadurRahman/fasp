# Project Context: faceswap

## Project Structure

```
face_swapper.py // actually swapface
main.py // uses face_swapper.py and utils.py 
utils.py // prosses images
```

## Project Files

### utils.py

```python
# utils.py
import cv2
import os

def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}. Check if it's a valid image file.")
    return img

def save_image(path, img):
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success = cv2.imwrite(path, img)
    if not success:
        raise ValueError(f"Failed to save image to {path}")
```

### face_swapper.py

```python
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
except ImportError:
    raise ImportError("InsightFace library is required. Install with: pip install insightface")

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("ONNX Runtime is required. Install with: pip install onnxruntime")

onnx_model = "inswapper_128.onnx"  # Default model

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

            self.swapper = get_model(model_name, download=True, download_zip=True)

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
                result_img = self.swapper.get(result_img, target_face, source_face, paste_back=True)
            except Exception as e:
                print(f"⚠️ Failed to swap one face: {str(e)}")
                continue

        return result_img

```

### main.py

```python
# main.py
import os
import time
import argparse
from face_swapper import FaceSwapper
from utils import load_image, save_image
from tqdm import tqdm
from pathlib import Path

# Enable verbose OpenVINO logs (optional)
os.environ["OPENVINO_VERBOSE"] = "1"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def is_image_file(path):
    return path.suffix.lower() in IMAGE_EXTS

def get_target_images(target_path):
    path = Path(target_path)
    if path.is_file() and is_image_file(path):
        return [path]
    elif path.is_dir():
        return sorted([p for p in path.glob("*") if is_image_file(p)])
    else:
        raise ValueError(f"Invalid target path: {target_path}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Face swap using InsightFace + ONNX")
    parser.add_argument("--source", required=True, help="Path to source image")
    parser.add_argument("--target", required=True, help="Path to target image or directory")
    parser.add_argument("--output", default="assets/output", help="Output directory path")
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")

    source_img = load_image(str(source_path))
    swapper = FaceSwapper()

    target_images = get_target_images(target_path)
    if not target_images:
        print("No valid target images found.")
        return

    print(f"✓ Found {len(target_images)} image(s) to process.")
    print("⚙️  Starting face swap...")

    # Time first image to estimate
    init_time = time.time()
    test_target_img = load_image(str(target_images[0]))
    _ = swapper.swap(test_target_img, source_img)
    est_time = time.time() - init_time
    total_est = est_time * len(target_images)
    print(f"⏱ Estimated total time: {total_est:.2f} seconds ({total_est/60:.1f} min)")

    for img_path in tqdm(target_images, desc="🔄 Swapping faces", unit="img"):
        try:
            target_img = load_image(str(img_path))
            result = swapper.swap(target_img, source_img)
            output_name = f"s_{source_path.stem}-{img_path.stem}{img_path.suffix}"
            output_path = output_dir / output_name
            save_image(str(output_path), result)
        except Exception as e:
            print(f"⚠️ Failed to process {img_path.name}: {e}")

    print("✅ Face swapping completed.")

if __name__ == "__main__":
    main()

```


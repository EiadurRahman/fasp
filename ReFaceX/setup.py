import subprocess
import os
import requests
import sys
import shutil
import platform


MODEL_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "inswapper_128.onnx")


def print_section(title):
    print(f"\n=== {title} ===")


def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_cuda_available():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


def is_intel_gpu():
    try:
        output = subprocess.check_output(["lspci"], stderr=subprocess.DEVNULL).decode()
        return "Intel" in output and "Graphics" in output
    except Exception:
        return False


def choose_onnxruntime():
    if is_colab() or is_cuda_available():
        return "onnxruntime-gpu"
    elif is_intel_gpu():
        return "onnxruntime-openvino"
    else:
        return "onnxruntime"


def pip_install(*packages, user_fallback=True):
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)
    except subprocess.CalledProcessError:
        if user_fallback:
            print("⚠️ Trying pip install with --user")
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", *packages], check=True)
        else:
            raise


def install_onnxruntime():
    runtime_pkg = choose_onnxruntime()
    print(f"📦 Installing ONNX Runtime: {runtime_pkg}")
    pip_install(runtime_pkg)


def install_ffmpeg():
    print_section("FFmpeg Check")
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("🎞️ FFmpeg is already installed.")
    except Exception:
        if is_colab():
            print("⚠️ FFmpeg is preinstalled in Google Colab.")
            return
        print("📦 Installing FFmpeg via apt...")
        subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"], check=True)


def install_base_dependencies():
    print_section("Installing Base Dependencies")
    base_deps = ["numpy", "opencv-python", "tqdm", "insightface"]
    pip_install(*base_deps)


def download_model():
    print_section("Downloading Model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"✅ Model already exists at: {MODEL_PATH}")
        return

    print(f"⬇️ Downloading model from Hugging Face...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)


def setup():
    print_section("🔧 ReFaceX Setup Starting")
    install_base_dependencies()
    install_onnxruntime()
    install_ffmpeg()
    download_model()
    print_section("✅ ReFaceX Setup Complete!")


if __name__ == "__main__":
    try:
        setup()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)

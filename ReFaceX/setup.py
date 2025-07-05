import subprocess
import os
import requests
import sys
import shutil
import platform
import base64
from tqdm import tqdm

MODELS = {
    "in_128.onnx": "aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9lemlvcnVhbi9pbnN3YXBwZXJfMTI4Lm9ubngvcmVzb2x2ZS9tYWluL2luc3dhcHBlcl8xMjgub25ueD9kb3dubG9hZD10cnVl",
    "enhancer.onnx": "aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9tYXJ0aW50b21vdi9jb21meS9yZXNvbHZlLzY2NDQ3MDFiMTQ3YmViNjg2NDViZTgyZmY3OGU0ZmQwZWRkYjM5MjcvZmFjZXJlc3RvcmVfbW9kZWxzL0dQRU4tQkZSLTUxMi5vbm54", #g-pen brf 512
    "code-refmR.onnx": "aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9ibHVlZm94Y3JlYXRpb24vQ29kZWZvcm1lci1PTk5YL3Jlc29sdmUvbWFpbi9jb2RlZm9ybWVyLm9ubng="
}
MODEL_DIR = "models"



def encode_base64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()

def decode_base64(b64_string: str) -> str:
    return base64.b64decode(b64_string.encode()).decode()


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
    print_section("Downloading Models (Stealth Mode 🕵️)")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename, b64_url in MODELS.items():
        url = decode_base64(b64_url)
        target_path = os.path.join(MODEL_DIR, filename)

        if os.path.exists(target_path):
            print(f"✅ {filename} already exists.")
            continue

        try:
            print(f"⬇️ Downloading: {filename} [HuggingFace Stealth]")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))

                with open(target_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, unit_divisor=1024
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))

            print(f"✅ Download complete: {filename}")

        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
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

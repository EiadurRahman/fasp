import subprocess
import os
import requests

def install_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("FFmpeg is already installed.")
    except subprocess.CalledProcessError:
        print("FFmpeg not found. Installing...")
        if subprocess.run(["sudo", "apt-get", "install", "-y", "ffmpeg"]).returncode == 0:
            print("FFmpeg installed successfully.")
        else:
            print("Failed to install FFmpeg. Please install it manually.")

def install_dependencies():
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")


def install_models():
    model_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true"
    model_dir = "models"
    model_path = os.path.join(model_dir, "inswapper_128.onnx")

    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        print(f"✅ Model already exists: {model_path}")
        return

    print("⬇️ Downloading inswapper_128.onnx...")
    response = requests.get(model_url, stream=True)

    if response.status_code == 200:
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"✅ Model downloaded to: {model_path}")
    else:
        print(f"❌ Failed to download model. Status code: {response.status_code}")

if __name__ == "__main__":
    install_models()
    # install_ffmpeg()
 
# prosses_video.py

import os
import shutil
import subprocess
from tqdm import tqdm
import cv2
from swap import FaceSwapper
import utils

TEMP_DIR = "temp"
INPUT_FRAMES_DIR = os.path.join(TEMP_DIR, "input_frames")
OUTPUT_FRAMES_DIR = os.path.join(TEMP_DIR, "output_frames")
AUDIO_PATH = os.path.join(TEMP_DIR, "audio.wav")

def clear_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(INPUT_FRAMES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

def extract_frames_and_audio(video_path):
    print("🎥 Extracting frames and audio...")
    clear_temp()

    # Extract frames
    subprocess.run([
        "ffmpeg", "-i", video_path,
        os.path.join(INPUT_FRAMES_DIR, "%06d.png")
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract audio (fail silently if no audio)
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a",
            AUDIO_PATH
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(AUDIO_PATH):
            print("⚠️ No audio found.")
    except Exception as e:
        print(f"⚠️ Audio extraction error: {e}")

def process_frames(swapper: FaceSwapper, source_face):
    frame_files = sorted([
        f for f in os.listdir(INPUT_FRAMES_DIR)
        if f.lower().endswith((".png", ".jpg"))
    ])

    print(f"🖼️ Processing {len(frame_files)} frame(s)...")

    for frame_file in tqdm(frame_files):
        input_path = os.path.join(INPUT_FRAMES_DIR, frame_file)
        output_path = os.path.join(OUTPUT_FRAMES_DIR, frame_file)

        try:
            img = utils.load_image(input_path)
            swapped = swapper.swap_one_face(img, source_face)
            utils.save_image(output_path, swapped)
        except Exception as e:
            print(f"⚠️ Skipping frame {frame_file}: {e}")

def merge_video(output_path, fps=25):
    print("🎬 Merging video...")

    input_pattern = os.path.join(OUTPUT_FRAMES_DIR, "%06d.png")

    if os.path.exists(AUDIO_PATH):
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern,
            "-i", AUDIO_PATH, "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", output_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", input_pattern,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path
        ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def cleanup():
    print(f'🧹 Cleaning up temporary files in {TEMP_DIR}...')
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    print("🧹 Temporary files cleaned up.")

def process_video(video_path, source_path, model_path, output_path, provider="cpu"):
    print(f"🎞️ Starting face swap on video: {video_path}")
    
    extract_frames_and_audio(video_path)

    # Load face swapper
    swapper = FaceSwapper(model_path=model_path, provider=provider)
    source_face = utils.process_source(source_path)

    process_frames(swapper, source_face)
    merge_video(output_path)

    print(f"✅ Output saved to: {output_path}")
    cleanup()

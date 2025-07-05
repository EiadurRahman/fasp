import os
import time
import argparse
from pathlib import Path
from swap import FaceSwapper
import utils
import img
import vid as process_video

SUPPORTED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv'}
SUPPORTED_SOURCE_EXT = {'.fsz', '.jpg', '.png', '.npz'}


def is_video_file(path):
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXT

def is_source_file(path):
    return Path(path).suffix.lower() in SUPPORTED_SOURCE_EXT

def process_video_file(video_path, source_path, model_path, output_path, provider):
    init_time = time.time()
    process_video.process_video(
        video_path, source_path, model_path, output_path, provider=provider, enhance=args.enhance
    )
    print(f"[✓] Processed video: {video_path}")
    print(f"    └─ Time taken: {time.time() - init_time:.2f}s")

def run(args):
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    source_path = Path(args.source).expanduser()

    assert input_path.exists(), f"Input not found: {input_path}"
    assert source_path.exists(), f"Source face file not found: {source_path}"
    assert is_source_file(source_path), f"Unsupported source format: {source_path.suffix}"

    if output_path.suffix == "":
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    source_face = utils.process_source(str(source_path))
    swapper = FaceSwapper(args.model, provider=args.provider)

    if input_path.is_file():
        if img.is_image_file(input_path):
            img.process_image_file(str(input_path), source_face, swapper, output_path,enhance=args.enhance)
        elif is_video_file(input_path):
            video_out = str(output_path) if output_path.suffix else str(output_path / f"swapped_{input_path.stem}.mp4")
            process_video_file(str(input_path), str(source_path), args.model, video_out, args.provider)
        else:
            print(f"[!] Unsupported file format: {input_path.suffix}")
    elif input_path.is_dir():
        img.process_images_in_directory(input_path, source_face, swapper, output_path, enhance=args.enhance)
        for file in input_path.iterdir():
            if is_video_file(file):
                video_out = output_path / f"swapped_{file.stem}.mp4"
                process_video_file(str(file), str(source_path), args.model, str(video_out), args.provider)
    else:
        print("[!] Input is neither a file nor a directory.")

    utils.clear_temp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Swapper CLI")
    parser.add_argument("--input", "-t", required=True, help="Path to image/video file or directory")
    parser.add_argument("--source", "-s", required=True, help="Source face (.jpg/.fsz/.npz)")
    parser.add_argument("--output", "-o", required=True, help="Output path or directory")
    parser.add_argument("--model", "-m", default="models/in_128.onnx", help="Path to ONNX model")
    parser.add_argument("--provider", "-p", default="cpu", choices=["cpu", "cuda", "openvino"], help="Execution provider")
    parser.add_argument("--enhance","-e", action="store_true", help="Apply GPEN post-processing to swapped face(s)")


    args = parser.parse_args()
    run(args)

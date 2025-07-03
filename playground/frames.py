import os
import subprocess


def extract_audio(video_path, output_audio_path):
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    print("🎧 Extracting audio...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        output_audio_path
    ]
    subprocess.run(cmd, check=True)
    print("✅ Audio extracted.")


def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("🖼️ Extracting frames...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        os.path.join(output_dir, "frame_%05d.png")
    ]
    subprocess.run(cmd, check=True)
    print("✅ Frames extracted.")


def stitch_video(frames_dir, audio_path, output_path="final_output.mp4", fps=25):
    print("🎬 Stitching video...")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Final video saved to: {output_path}")

def clean_file_name(file_dir,string_remove):
    files = os.listdir(file_dir)
    for file_name in files:
        file_name = os.path.join(file_dir, file_name)
        if not os.path.isfile(file_name):
            continue
        if string_remove in file_name:
            print(f"Cleaning file name: {file_name}")
            new_file_name = file_name.replace(string_remove, "")
            os.rename(file_name, new_file_name)
        
        print("✅ File names cleaned.")

def conver_jpeg_to_png(file_dir):
    files = os.listdir(file_dir)
    for file_name in files:
        file_name = os.path.join(file_dir, file_name)
        if not os.path.isfile(file_name):
            continue
        if file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
            print(f"Converting {file_name} to PNG...")
            new_file_name = file_name.rsplit('.', 1)[0] + ".png"
            cmd = ["convert", file_name, new_file_name]
            subprocess.run(cmd, check=True)
            os.remove(file_name)  # Remove the original JPEG file
            print(f"✅ Converted {file_name} to {new_file_name}.")


if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")

    input_video = "/home/eiadurrahman/ehm/trgt/0090.avi"
    audio_out = os.path.join("temp", "audio.wav")
    input_frames_dir = os.path.join("temp", "input_frames")
    output_frames_dir = "temp/output_frames"
    final_output = os.path.join("temp", "final_output.mp4")

    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        extract_audio(input_video, audio_out)
    except:
        print('no audio')
    extract_frames(input_video, input_frames_dir)
    # clean_file_name(output_frames_dir, "s_2-compressed_")
    # conver_jpeg_to_png(output_frames_dir)
    # stitch_video(output_frames_dir, audio_out, output_path=final_output)

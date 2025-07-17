import cv2
import os
import numpy as np
from utils import face_detected, save_image

def frame_difference(img1, img2, threshold=30):
    """
    Compute a simple frame difference score between two images.
    Returns True if the images are different enough.
    """
    if img1 is None or img2 is None:
        return True

    # Resize to same shape
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute mean absolute difference
    diff = np.mean(cv2.absdiff(gray1, gray2))

    return diff > threshold

def extract_faces_from_video(video_path, output_dir, max_frames=None, diff_threshold=30, frame_skip=3):
    """
    Extract frames from a video that contain faces and are visually distinct.
    
    Parameters:
        - video_path: path to video file
        - output_dir: directory to save selected frames
        - max_frames: optional max number of frames to save
        - diff_threshold: how different two frames must be to be saved
        - frame_skip: number of frames to skip between extractions
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    prev_frame = None
    saved = 0
    frame_id = 0

    print(f"📼 Processing video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            if face_detected(frame):
                if frame_difference(prev_frame, frame, threshold=diff_threshold):
                    filename = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
                    save_image(filename, frame)
                    prev_frame = frame.copy()
                    saved += 1
                    print(f"✅ Saved frame {frame_id}")
                else:
                    print(f"🟡 Skipped similar frame {frame_id}")
            else:
                print(f"❌ No valid face in frame {frame_id}")
        
        frame_id += 1
        if max_frames and saved >= max_frames:
            print(f"🔻 Reached max frame limit: {max_frames}")
            break

    cap.release()
    print(f"🎉 Extraction complete. Saved {saved} frames to {output_dir}")


if __name__ == "__main__":          
    extract_faces_from_video(
        video_path="/home/eiadurrahman/ayo/mrm/mrim.mp4",
        output_dir="/home/eiadurrahman/ayo/mrm/imgs",
        diff_threshold=30,
        frame_skip=3
    )
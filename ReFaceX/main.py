import utils
import os
import time
from swap import FaceSwapper

target_image_path = "/home/eiadurrahman/Desktop/faceswap/files/imgs/aish2.jpg"
source_path = "/home/eiadurrahman/Desktop/faceswap/files/nehal.fsz"  # or .jpg or .npz

model_path = "models/inswapper_128.onnx"

# Load data
target_img = utils.load_image(target_image_path)
source_face = utils.process_source(source_path)

# Initialize swapper
swapper = FaceSwapper(model_path, provider="cpu")
init_time = time.time()
swapped = swapper.swap_one_face(target_img, source_face)
print(f"Swapping completed in {time.time() - init_time:.2f} seconds")

utils.save_image(f"output/swapped_{os.path.basename(target_image_path)}", swapped)

# import vid as prosses_video

# video_path = "/home/eiadurrahman/Desktop/faceswap/files/vids/vid.mp4"
# # source_path = "faceset.fsz"
# output_path = "output/vid_swapped.mp4"

# init_time = time.time()
# prosses_video.process_video(video_path, source_path, model_path, output_path, provider="cpu")
# print(f"Video processing completed in {time.time() - init_time:.2f} seconds")

utils.clear_temp()
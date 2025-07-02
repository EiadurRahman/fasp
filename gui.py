# app.py
import gradio as gr
import cv2
import os
import time
from pathlib import Path
import zipfile
import shutil
import threading # For potentially long operations that might block the UI, though Gradio handles most with generators
import argparse

# Import your core logic
from face_swapper import FaceSwapper
from utils import load_image, save_image

# Enable verbose OpenVINO logs (optional)
os.environ["OPENVINO_VERBOSE"] = "1"

# --- Constants and Global Variables ---
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
GUI_OUTPUT_DIR = Path("assets/gui_output") # Dedicated output directory for GUI results

# Global list to store logs for the UI
processing_logs = []
# Initialize FaceSwapper once, potentially with a default provider.
# The provider can be selected in face_swapper.py directly or passed here.
# For simplicity and to use the 'provider' variable in face_swapper.py, we'll
# let face_swapper.py handle its own provider setup.
swapper = None # Initialize lazily to avoid potential issues during app startup

# --- Helper Functions (adapted from your main.py and utils.py) ---
def is_image_file(path_obj):
    """Checks if a Path object points to a valid image file."""
    return path_obj.suffix.lower() in IMAGE_EXTS

def get_target_images_paths(target_input):
    """
    Determines the list of target image paths from Gradio's file input.
    Can handle single file or multiple files/directory upload.
    """
    if isinstance(target_input, list): # Multiple files or directory
        image_paths = []
        for gr_file in target_input:
            p = Path(gr_file.name) # Gradio File object has a 'name' attribute for its path
            if p.is_file() and is_image_file(p):
                image_paths.append(p)
            elif p.is_dir(): # If a directory was uploaded, iterate its contents
                for img_p in sorted([f for f in p.glob("*") if is_image_file(f)]):
                    image_paths.append(img_p)
        return image_paths
    elif isinstance(target_input, str): # Single file
        p = Path(target_input)
        if p.is_file() and is_image_file(p):
            return [p]
    return [] # Return empty list if no valid images found

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def append_log(message):
    """Appends a timestamped message to the global logs and returns the formatted string."""
    global processing_logs
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    processing_logs.append(f"[{timestamp}] {message}")
    return "\n".join(processing_logs)

def clear_logs():
    """Clears the global logs."""
    global processing_logs
    processing_logs = []
    return ""

# --- Gradio UI Functions ---

def initialize_swapper_if_needed():
    """Initializes the FaceSwapper instance if it hasn't been already."""
    global swapper
    if swapper is None:
        try:
            swapper = FaceSwapper()
            append_log("FaceSwapper initialized successfully.")
        except Exception as e:
            append_log(f"Error initializing FaceSwapper: {e}. Please check dependencies and console logs.")
            swapper = None # Ensure it remains None on failure
    return "\n".join(processing_logs)

def preview_swap(source_file, target_files):
    """
    Performs a single face swap for preview.
    Uses the first image from the target selection if a directory or multiple files are chosen.
    """
    log_output = clear_logs()
    log_output = append_log("Generating preview...")
    yield None, log_output # Clear previous preview and update logs

    log_output = initialize_swapper_if_needed()
    if swapper is None:
        yield None, log_output # Return if swapper failed to initialize

    if not source_file:
        log_output = append_log("Error: Source image not provided for preview.")
        yield None, log_output
        return
    if not target_files: # target_files will be a list if it's a File component with file_count
        log_output = append_log("Error: Target image(s) not provided for preview.")
        yield None, log_output
        return

    source_path = Path(source_file.name)
    target_paths = get_target_images_paths(target_files)

    if not target_paths:
        log_output = append_log("Error: No valid target images found for preview.")
        yield None, log_output
        return

    # For preview, always use the first detected target image
    target_img_path_for_preview = target_paths[0]

    try:
        source_img = load_image(str(source_path))
        target_img = load_image(str(target_img_path_for_preview))
        
        result_img = swapper.swap(target_img, source_img)
        
        # --- ADD THIS LINE FOR RGB CONVERSION FOR DISPLAY ---
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        # ---------------------------------------------------

        log_output = append_log(f"Preview generated using {target_img_path_for_preview.name}.")
        yield result_img_rgb, log_output # Yield the RGB converted image
    except Exception as e:
        log_output = append_log(f"Error during preview: {e}")
        yield None, log_output

def start_face_swap_process(source_file, target_files, progress=gr.Progress(track_tqdm=True)):
    """
    Starts the batch face swap process.
    Updates logs and progress bar.
    """
    log_output = clear_logs()
    log_output = append_log("Starting face swap process...")
    # Yield initial state: clear preview, update logs, disable download button
    yield None, log_output, gr.Button("Zipping...", interactive=False)

    log_output = initialize_swapper_if_needed()
    if swapper is None:
        yield None, log_output, gr.Button("Download Zip", interactive=False) # Keep disabled on error
        return

    if not source_file:
        log_output = append_log("Error: Source image not provided.")
        yield None, log_output, gr.Button("Download Zip", interactive=False)
        return

    source_path = Path(source_file.name)
    ensure_dir(GUI_OUTPUT_DIR) # Ensure the output directory exists

    target_images_to_process = get_target_images_paths(target_files)
    
    if not target_images_to_process:
        log_output = append_log("No valid target images found to process.")
        yield None, log_output, gr.Button("Download Zip", interactive=False)
        return

    log_output = append_log(f"Found {len(target_images_to_process)} image(s) to process.")
    
    # Load source image once
    try:
        source_img = load_image(str(source_path))
    except Exception as e:
        log_output = append_log(f"Error loading source image: {e}")
        yield None, log_output, gr.Button("Download Zip", interactive=False)
        return

    append_log("⚙️ Starting face swap...")

    # For better responsiveness, we update logs more frequently
    # The progress bar is handled by `gr.Progress(track_tqdm=True)`
    for img_path in progress.tqdm(target_images_to_process, desc="🔄 Swapping faces", unit="img"):
        try:
            target_img = load_image(str(img_path))
            result = swapper.swap(target_img, source_img)
            
            output_name = f"s_{source_path.stem}-{img_path.stem}{img_path.suffix}"
            output_path = GUI_OUTPUT_DIR / output_name
            save_image(str(output_path), result)
            log_output = append_log(f"Processed {img_path.name} -> {output_name}")
            yield None, log_output, gr.Button("Download Zip", interactive=False) # Keep disabled during process
        except Exception as e:
            log_output = append_log(f"⚠️ Failed to process {img_path.name}: {e}")
            yield None, log_output, gr.Button("Download Zip", interactive=False) # Update logs even on error

    log_output = append_log("✅ Face swapping completed.")
    # Finally, enable the download button
    yield None, log_output, gr.Button("Download Zip", interactive=True)


def download_results_zip():
    """Zips the output directory and returns the path to the zip file for download."""
    log_output = append_log("File zipping started, please wait...")
    yield None, log_output # Update log

    zip_base_name = "face_swapped_results"
    zip_path = str(GUI_OUTPUT_DIR.parent / zip_base_name) # Zip to assets/face_swapped_results.zip

    try:
        shutil.make_archive(zip_path, 'zip', GUI_OUTPUT_DIR)
        final_zip_file = f"{zip_path}.zip"
        log_output = append_log(f"Zipping complete! Download link available: {Path(final_zip_file).name}")
        
        # Clean up the output directory after zipping
        if GUI_OUTPUT_DIR.exists():
            shutil.rmtree(GUI_OUTPUT_DIR)
            ensure_dir(GUI_OUTPUT_DIR) # Recreate empty directory for next run
            log_output = append_log("Output directory cleaned.")
        
        yield final_zip_file, log_output
    except Exception as e:
        log_output = append_log(f"Error during zipping: {e}")
        yield None, log_output # No file to download on error

# --- Gradio UI Layout ---
with gr.Blocks(title="Face Swap GUI") as demo:
    gr.Markdown(
        """
        # 🚀 Face Swap GUI
        Swap faces in images using your local models!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Step 1: Upload Images", open=True):
                # 1. Input Fields: Source and Target File/Directory
                source_image_input = gr.File(
                    label="Upload Source Face Image",
                    type="filepath", # Provides the path to the temporary file
                    file_count="single",
                    file_types=[f".{ext.strip('.')}" for ext in IMAGE_EXTS]
                )
                target_input = gr.File(
                    label="Upload Target Image(s) or Directory",
                    file_count="multiple", # Allows single file, multiple files, or a directory
                    type="filepath",
                    file_types=[f".{ext.strip('.')}" for ext in IMAGE_EXTS]
                )
                gr.Markdown("*(If you upload a directory, the first image found will be used for preview.)*")

            with gr.Accordion("Step 2: Actions", open=True):
                with gr.Row():
                    # 3. Preview Button
                    preview_button = gr.Button("🖼️ Preview Swap (Single)", size="lg")
                    # 2. Start Button
                    start_button = gr.Button("🚀 Start Batch Swap", size="lg")
                
                # 5. Download Zip Button
                download_zip_button = gr.Button("💾 Download Swapped Images (ZIP)", interactive=False)
                gr.Markdown("*(This button becomes active after batch processing completes)*")

        with gr.Column(scale=2):
            # 3. Preview Output
            preview_output = gr.Image(label="Preview of Swapped Face", type="numpy", height=400)
            
            # 4. Progress Bar (Gradio automatically adds this when `gr.Progress` is used in the function)
            
            # 6. View Logs
            process_log_output = gr.Textbox(
                label="Process Logs", 
                lines=15, 
                autoscroll=True, 
                interactive=False, 
                elem_id="logs_box"
            )
            gr.ClearButton([source_image_input, target_input, preview_output, process_log_output])

    # --- Event Handlers ---

    # Initialize swapper when the app loads (hidden call)
    demo.load(initialize_swapper_if_needed, inputs=[], outputs=[process_log_output])

    preview_button.click(
        fn=preview_swap,
        inputs=[source_image_input, target_input],
        outputs=[preview_output, process_log_output]
    )

    start_button.click(
        fn=start_face_swap_process,
        inputs=[source_image_input, target_input],
        outputs=[preview_output, process_log_output, download_zip_button]
    )

    download_zip_button.click(
        fn=download_results_zip,
        inputs=[],
        outputs=[gr.File(label="Download", file_count="single", type="filepath"), process_log_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="use --live to make it run in live mode")
    parser.add_argument("--live", action="store_true", help="Run the app in live mode")
    args = parser.parse_args()

    if args.live:
        live = True
    else:
        live = False
    demo.launch(share=live,server_port=1313)

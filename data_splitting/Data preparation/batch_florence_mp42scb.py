import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from decord import VideoReader, cpu
from tqdm import tqdm

# # Initialize Decord video reader
# decord.bridge.set_bridge("torch")

# Define parameters
batch_size = 64  # Number of frames to process in one batch
model_id = 'microsoft/Florence-2-base-ft'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load OCR model and processor
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision='refs/pr/6'
).to(device)
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    revision='refs/pr/6'
)

# Define batch OCR function
def run_ocr_batch(frames, task_prompt="<OCR>"):
    """
    Runs OCR on a batch of frames.
    """
    with torch.no_grad():  # Ensure no gradients are stored
        inputs = processor(
            text=[task_prompt] * len(frames),
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(device)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        # Move tensors back to CPU to free GPU memory
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)
        parsed_answers = [
            processor.post_process_generation(
                answer, task=task_prompt, image_size=(frames[i].width, frames[i].height)
            )
            for i, answer in enumerate(generated_text)
        ]

        # Clean up GPU memory
        del inputs, generated_ids
        torch.cuda.empty_cache()  # Force free memory
        return parsed_answers

# Main processing function
def process_videos(source_dir, dest_dir):
    """
    Processes all .mp4 files in the source directory, performs batch OCR on frames, 
    and saves results in JSON format in the destination directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    video_files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]

    for video_file in tqdm(video_files):
        video_path = os.path.join(source_dir, video_file)
        output_file = os.path.join(dest_dir, f"{os.path.splitext(video_file)[0]}_scb.json")
        
        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {video_file}: Output file already exists.")
            continue

        print(f"Processing video: {video_file}...")
        try:
            # Read the video
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_rate = int(vr.get_avg_fps())
            duration = len(vr) / frame_rate

            # Extract frames at one-second intervals
            ocr_results = []
            frames = []
            times = []
            for second in tqdm(range(0, int(duration))):
                frame_idx = second * frame_rate
                if frame_idx >= len(vr):
                    break
                
                frame = vr[frame_idx].asnumpy()
                frame_image = Image.fromarray(frame)
                frames.append(frame_image)
                times.append(second)

                # Process batch when it reaches the batch size
                if len(frames) == batch_size:
                    #print(f"Running OCR on batch of size {batch_size}...")
                    ocr_texts = run_ocr_batch(frames)
                    for time, text in zip(times, ocr_texts):
                        ocr_results.append({"time": time, "scb": text})
                    frames = []
                    times = []

            # Process remaining frames
            if frames:
                print(f"Running OCR on final batch of size {len(frames)}...")
                ocr_texts = run_ocr_batch(frames)
                for time, text in zip(times, ocr_texts):
                    ocr_results.append({"time": time, "scb": text})

            # Save results to a JSON file
            with open(output_file, "w") as f:
                json.dump(ocr_results, f, indent=4)

            print(f"Results saved to {output_file}")
        
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

        # Force clean GPU memory at the end of each video
        torch.cuda.empty_cache()


# Example usage
source_dir = "/home/ACG/data/raw/videos"
dest_dir = "/home/ACG/data/raw/scoreboards"
process_videos(source_dir, dest_dir)

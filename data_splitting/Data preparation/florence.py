import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import decord
import pickle
import os

# Initialize Decord video reader
decord.bridge.set_bridge("torch")
from decord import VideoReader, cpu

# Load OCR model and processor
model_id = 'microsoft/Florence-2-base-ft'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Define the OCR function
def run_ocr_on_frame(frame_image, task_prompt="<OCR>"):
    """
    Runs the OCR model on a single frame image.
    """
    inputs = processor(text=task_prompt, images=frame_image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(frame_image.width, frame_image.height)
    )
    return parsed_answer

# Read video and process frames
video_path = "/home/ACG/data/raw/test.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

vr = VideoReader(video_path, ctx=cpu(0))  # Use CPU context for Decord
frame_count = min(100, len(vr))  # Limit to the first 100 frames

ocr_results = {}  # Dictionary to store OCR results
for i in range(frame_count):
    frame = vr[i].detach().numpy()  # Convert frame to NumPy array
    frame_image = Image.fromarray(frame)  # Convert to PIL image
    
    print(f"Processing frame {i+1}/{frame_count}...")
    ocr_result = run_ocr_on_frame(frame_image)
    ocr_results[i] = ocr_result

# Save the results as a pickle file
output_path = "/home/ACG/data/ocr_results.pkl"
with open(output_path, "wb") as f:
    pickle.dump(ocr_results, f)

print(f"OCR results saved to {output_path}")

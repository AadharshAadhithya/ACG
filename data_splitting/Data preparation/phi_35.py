import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import decord
import pickle
import os
import time 


# Set up Decord for video frame reading
decord.bridge.set_bridge("torch")
from decord import VideoReader, cpu

# Model and processor initialization
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

# Read video frames
video_path = "/home/ACG/data/raw/test.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

vr = VideoReader(video_path, ctx=cpu(0))  # Use CPU context for Decord
frame_count = min(5, len(vr))  # Limit to the first 100 frames

# Prepare images and placeholders for a single batch query
images = []  # To store all frame images
placeholders = ""  # To store all placeholders for the query

for i in range(frame_count):
    frame = vr[i].detach().numpy()  # Convert frame to NumPy array
    frame_image = Image.fromarray(frame)  # Convert to PIL image
    images.append(frame_image)  # Append the image
    placeholders += f"<|image_{i+1}|>\n"  # Append the corresponding placeholder


prompt = """
From the Image give the score board in JSON format. Strictly follow the below JSON format. 
\{ runs: , overs:, batsman_1: {runs: , balls:}, batsman_2: {runs: balls:}, min_remaining:  \}
"""
# Create a single batch query with all frames
messages = [
    {"role": "user", "content": placeholders +prompt  },
]

prompt = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Process the batch input
inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False,
}
t = time.time()
generate_ids = model.generate(
    **inputs,
    eos_token_id=processor.tokenizer.eos_token_id,
    **generation_args
)
e= time.time()

print(f"time taken:{e-t} ")

# Remove input tokens from the generated IDs
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

# Parse the results
ocr_results = {}
response_lines = response.split("\n")

for i, line in enumerate(response_lines):
    ocr_results[i] = line.strip()
    
    
print(ocr_results)

# Save results to a pickle file
output_path = "/home/ACG/data/ocr_results_phi3.5.pkl"
with open(output_path, "wb") as f:
    pickle.dump(ocr_results, f)

print(f"OCR results saved to {output_path}")

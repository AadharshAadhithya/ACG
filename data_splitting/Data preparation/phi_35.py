import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import decord
import pickle
import os

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

# Function to run OCR on a single frame
def run_ocr_on_frame(image, frame_idx):
    placeholder = f"<|image_{frame_idx}|>\n"
    messages = [
        {"role": "user", "content": placeholder + "Perform OCR on the image."},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )
    # Remove input tokens from the generated IDs
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return response

# Read video frames
video_path = "/home/ACG/data/raw/test.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

vr = VideoReader(video_path, ctx=cpu(0))  # Use CPU context for Decord
frame_count = min(100, len(vr))  # Limit to the first 100 frames

# Process each frame and store the results
ocr_results = {}
for i in range(frame_count):
    frame = vr[i].detach().numpy()  # Convert frame to NumPy array
    frame_image = Image.fromarray(frame)  # Convert to PIL image

    print(f"Processing frame {i+1}/{frame_count}...")
    ocr_result = run_ocr_on_frame(frame_image, i + 1)
    ocr_results[i] = ocr_result

# Save results to a pickle file
output_path = "/home/ACG/data/ocr_results_phi3.5.pkl"
with open(output_path, "wb") as f:
    pickle.dump(ocr_results, f)

print(f"OCR results saved to {output_path}")

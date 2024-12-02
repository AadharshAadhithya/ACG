import os
import glob
import torch
from torchvision import transforms
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
from clip_vit import create_clip_vit_L

# Set frame sampling rate (1 frame per 1 second)
frame_rate = 0.5  # 1 frame every second

# Batch size for inference
batch_size = 64  # Adjust this value based on available GPU memory

# Function to save embeddings
def save_embeddings(embeddings, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    np.save(save_path, embeddings)

# Load the model
model = create_clip_vit_L(precision="fp32")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Directory containing video files
video_files = glob.glob('/home/ACG/data/raw/videos/*.mp4')
print(video_files)

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

# Directory to save embeddings
output_dir = '/home/ACG/data/pre_2/vid_embs'

# Loop through each video file
for video_file in tqdm(video_files):
    try:
        # Load video
        vr = VideoReader(video_file, ctx=cpu(0))
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()

        # Sample frames at a rate of 1 frame every `frame_rate` seconds
        sampled_indices = list(range(0, int(total_frames), int(frame_rate * avg_fps)))
        frames = vr.get_batch(sampled_indices).asnumpy()

        # Preprocess all frames for the video
        preprocessed_frames = torch.stack([transform(frame) for frame in frames])
        preprocessed_frames = preprocessed_frames.to(device)

        # Initialize a list to collect embeddings
        embeddings_list = []

        # If the total number of preprocessed frames is greater than batch_size
        if len(preprocessed_frames) > batch_size:
            for i in range(0, len(preprocessed_frames), batch_size):
                batch_frames = preprocessed_frames[i:i + batch_size]
                with torch.no_grad():
                    batch_embeddings = model(batch_frames)
                embeddings_list.append(batch_embeddings.detach().cpu().numpy())
        else:
            # If total frames are less than or equal to batch_size, process them all at once
            with torch.no_grad():
                batch_embeddings = model(preprocessed_frames)
            embeddings_list.append(batch_embeddings.detach().cpu().numpy())

        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings_list, axis=0)
        


        # Save the embeddings to the output directory
        video_name = os.path.basename(video_file)  # Get video file name
        save_path = os.path.join(output_dir, os.path.splitext(video_name)[0] + '_embeddings.npy')
        save_embeddings(embeddings, save_path)

    except Exception as e:
        print("------------------")
        print(f"Error processing {video_file}")
        print(e)
        print("------------------")

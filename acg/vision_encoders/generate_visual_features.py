import os
import glob
import torch
from torchvision import transforms
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm
from clip_vit import create_clip_vit_L

# Parameter to control the frame sampling rate
frame_rate = 2  # Frame sampling rate in seconds (1 frame per 2 seconds)


def save_embeddings(embeddings, save_path):
    np.save(save_path, embeddings)


model = create_clip_vit_L(precision="fp32")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

video_files = glob.glob('/home/ACG/data/pre_1/*.mp4')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()


for video_file in tqdm(video_files):

        # Load video
        vr = VideoReader(video_file, ctx=cpu(0))
        total_frames = len(vr)
        
        # Sample frames at a rate of 1 frame every `frame_rate` seconds
        sampled_indices = list(range(0, int(total_frames), int(frame_rate * vr.get_avg_fps())))  # Adjust based on video FPS
        frames = vr.get_batch(sampled_indices).asnumpy()

        # Preprocess frames
        preprocessed_frames = torch.stack([transform(frame) for frame in frames])
        preprocessed_frames = preprocessed_frames.to(device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model(preprocessed_frames)
        
        # Save the embeddings to a .npy file
        save_path = os.path.splitext(video_file)[0] + '_embeddings.npy'
        save_embeddings(embeddings.cpu().numpy(), save_path)
    # except Exception as e:
    #     print("------------------")
    #     print(f"error proceessing {video_file}")
    #     print(e)
    #     print("------------------")
    
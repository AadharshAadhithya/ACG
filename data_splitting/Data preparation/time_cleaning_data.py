import os
import numpy as np
from tqdm import tqdm

# Paths to the dataset directories
base_data_dir = '/home/ACG/data/pre_3'  # Base directory containing train, val, and test

# Directories to be processed
directories = ['train', 'val', 'test']

# Function to delete files where time T > 200 and corresponding transcription files
def clean_large_time_embeddings(data_dir):
    vid_embs_dir = os.path.join(data_dir, 'vid_embs')
    transcriptions_dir = os.path.join(data_dir, 'transcriptions')

    # Get all .npy files from vid_embs directory
    files = [f for f in os.listdir(vid_embs_dir) if f.endswith('_embeddings.npy')]

    # Use tqdm for a progress bar
    for filename in tqdm(files, desc="Processing files", unit="file"):
        file_path = os.path.join(vid_embs_dir, filename)
        
        try:
            # Load the .npy file
            emb_array = np.load(file_path)
            
            # Get the first dimension (time T)
            T = emb_array.shape[0]
            
            # If T is greater than 200, delete the file and corresponding transcription
            if T > 200:
                print(f"Deleting file with time T > 200: {file_path}")
                os.remove(file_path)  # Use os.remove to delete the embedding file
                
                # Corresponding transcription file
                transcription_file = filename.replace('_embeddings.npy', '.txt')
                transcription_path = os.path.join(transcriptions_dir, transcription_file)
                
                if os.path.exists(transcription_path):
                    print(f"Deleting corresponding transcription file: {transcription_path}")
                    os.remove(transcription_path)  # Use os.remove to delete the transcription file
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Run the function for train, val, and test directories
for directory in directories:
    data_dir = os.path.join(base_data_dir, directory)
    clean_large_time_embeddings(data_dir)

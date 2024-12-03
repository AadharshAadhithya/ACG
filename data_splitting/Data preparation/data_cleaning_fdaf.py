import os
import numpy as np

# Paths to the dataset directories
base_data_dir = '/home/ACG/data/pre_3'  # Base directory containing train, val, and test

# Directories to be processed
directories = ['train', 'val', 'test']

# Function to delete empty embeddings and corresponding transcription files
def clean_empty_embeddings(data_dir):
    vid_embs_dir = os.path.join(data_dir, 'vid_embs')
    transcriptions_dir = os.path.join(data_dir, 'transcriptions')

    # Iterate over all files in the vid_embs directory
    for filename in os.listdir(vid_embs_dir):
        if filename.endswith('_embeddings.npy'):
            file_path = os.path.join(vid_embs_dir, filename)
            
            try:
                # Load the .npy file and check if it's empty
                emb_array = np.load(file_path)
                
                # If the array is empty, delete the file and its corresponding transcription
                if emb_array.size == 0:
                    print(f"Deleting empty file: {file_path}")
                    os.remove(file_path)
                    
                    # Corresponding transcription file
                    transcription_file = filename.replace('_embeddings.npy', '.txt')
                    transcription_path = os.path.join(transcriptions_dir, transcription_file)
                    
                    if os.path.exists(transcription_path):
                        print(f"Deleting corresponding transcription file: {transcription_path}")
                        os.remove(transcription_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Run the function for train, val, and test directories
for directory in directories:
    data_dir = os.path.join(base_data_dir, directory)
    clean_empty_embeddings(data_dir)

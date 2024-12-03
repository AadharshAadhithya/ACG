import os
import shutil

# Define the directories
transcripts_dir = '/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_2/standardized_transcripts'
vid_embs_dir = '/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_2/vid_embs'
output_transcripts_dir = '/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/standardized_transcripts_filtered'
output_vid_embs_dir = '/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/vid_embs'

# Create output directories if they don't exist
os.makedirs(output_transcripts_dir, exist_ok=True)
os.makedirs(output_vid_embs_dir, exist_ok=True)

# Function to process files
def process_files():
    # Loop through each file in the transcripts directory
    for filename in os.listdir(transcripts_dir):
        if filename.endswith('.txt'):
            txt_filepath = os.path.join(transcripts_dir, filename)
            
            # Read the content of the transcript file
            with open(txt_filepath, 'r') as f:
                content = f.read().strip().lower()

            # Check if the file contains "none" or variants
            if "none" in content:
                print(f"Skipping {filename} due to 'None' content.")
                continue  # Skip if the content is "None" or its variants

            # Construct the corresponding .npy filename
            npy_filename = filename.split('.txt')[0]
            npy_filename = npy_filename+"_embeddings.npy"
            
            npy_filepath = os.path.join(vid_embs_dir, npy_filename)

            # Check if the corresponding .npy file exists
            if os.path.exists(npy_filepath):
                
                # Copy the .txt and .npy files to the new directories
                shutil.copy(txt_filepath, output_transcripts_dir)
                shutil.copy(npy_filepath, output_vid_embs_dir)
                print(f"Copied {filename} and {npy_filename} to pre_3 directories.")
            else:
                print(npy_filepath)
                print(f"Corresponding .npy file not found for {filename}. Skipping.")

# Run the process
process_files()

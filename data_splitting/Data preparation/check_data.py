import os
import numpy as np
from tqdm import tqdm

# Define the expected shape
expected_shape = (None, None, None)  # TxPxD: flexible None for any T, P, D dimensions

# Directory containing .npy files
directory_path = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/test/vid_embs"  # Replace with your actual path

# Iterate through all .npy files in the directory
for file in tqdm(os.listdir(directory_path), desc="Processing .npy files"):
    file_path = os.path.join(directory_path, file)
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Check if the shape matches the expected format (TxPxD)
        if len(data.shape) != 3:  # Ensure it's a 3D array
            print(f"Deleting {file}: Incorrect shape {data.shape}")
            os.remove(file_path)
    except Exception as e:
        # If any error occurs during loading, delete the file
        print(f"Deleting {file}: Error while loading - {e}")
        os.remove(file_path)

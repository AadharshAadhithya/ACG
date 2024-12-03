import os
import numpy as np
import random
import shutil

# Directories
text_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/standardized_transcripts_filtered"
embs_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/vid_embs"

train_text_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/train/transcriptions"
train_embs_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/train/vid_embs"


val_text_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/val/transcriptions"
val_embs_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/val/vid_embs"


test_text_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/test/transcriptions"
test_embs_dir = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/pre_3/test/vid_embs"

# Create directories if they don't exist
for dir_path in [train_text_dir, train_embs_dir, val_text_dir, val_embs_dir, test_text_dir, test_embs_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Get text filenames
text_files = sorted([f for f in os.listdir(text_dir) if f.endswith(".txt")])

# Pair files if a match exists
file_pairs = []
for text_file in text_files:
    base_name = text_file.replace(".txt", "")
    emb_file = f"{base_name}_embeddings.npy"
    if emb_file in os.listdir(embs_dir):
        file_pairs.append((text_file, emb_file))

# Shuffle the pairs
random.seed(42)  # For reproducibility
random.shuffle(file_pairs)

# Splitting logic
total_files = len(file_pairs)
test_count = int(0.2 * total_files)
train_count = total_files - test_count
val_count = int(0.2 * train_count)

train_files = file_pairs[:train_count - val_count]
val_files = file_pairs[train_count - val_count:train_count]
test_files = file_pairs[train_count:]

# Function to move files
def move_files(file_list, text_src, embs_src, text_dst, embs_dst):
    for text_file, emb_file in file_list:
        shutil.move(os.path.join(text_src, text_file), os.path.join(text_dst, text_file))
        shutil.move(os.path.join(embs_src, emb_file), os.path.join(embs_dst, emb_file))

# Move the files to respective directories
move_files(train_files, text_dir, embs_dir, train_text_dir, train_embs_dir)
move_files(val_files, text_dir, embs_dir, val_text_dir, val_embs_dir)
move_files(test_files, text_dir, embs_dir, test_text_dir, test_embs_dir)

print("Files have been processed, matched, and moved to train, val, and test sets.")

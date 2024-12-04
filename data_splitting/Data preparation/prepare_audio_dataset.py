import os
import json
import pandas as pd
from tqdm import tqdm

# Directories containing audio files and transcripts
audio_dir = "/home/ACG/data/raw/short_audios"
transcript_dir = "/home/ACG/data/raw/whisperx_transcripts"
output_csv = "/home/ACG/data/raw/short_audios/metadata.csv"

# Initialize a list to hold data for the DataFrame
data = []

# Iterate over audio files
for audio_file in tqdm(os.listdir(audio_dir)):
    if audio_file.endswith(".mp3"):
        # Extract id (filename without extension)
        file_id = os.path.splitext(audio_file)[0]
        
        # Get corresponding JSON file
        json_file = os.path.join(transcript_dir, f"{file_id}.json")
        if os.path.exists(json_file):
            # Read the JSON file
            with open(json_file, "r", encoding="utf-8") as jf:
                transcript_data = json.load(jf)
                # Collect all text fields
                transcription = " ".join(segment["text"].strip() for segment in transcript_data)
                # Append data to the list
                data.append({
                    "id": file_id,
                    "file_name": os.path.join(audio_dir, audio_file),
                    "transcription": transcription
                })

# Create a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv(output_csv, index=False)

print(f"Metadata CSV saved to {output_csv}")

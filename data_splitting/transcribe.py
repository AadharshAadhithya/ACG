# from faster_whisper import WhisperModel
# import torch

# # Load the model
# model = WhisperModel("small.en", device="cuda" if torch.cuda.is_available() else "cpu")

# # Path to the audio file
# audio_path = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/output_audio.mp3"

# # Perform transcription
# segments, info = model.transcribe(audio_path, beam_size=5, language="en")

# # Print results
# for segment in segments:
#     print(f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text}")


import os
import argparse
from faster_whisper import WhisperModel
import ffmpeg
from tqdm import tqdm 

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000, format='wav').run(overwrite_output=True, quiet=True)
    except ffmpeg.Error as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None
    return audio_path

# Function to transcribe audio using Faster Whisper
def transcribe_audio(audio_path, model):
    segments, _ = model.transcribe(audio_path)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

# Main function
def process_videos(directory, model_size):
    # Load the Faster Whisper model
    print(f"Loading Faster Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Iterate through all .mp4 files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(directory, filename)
            audio_path = os.path.join(directory, filename.replace(".mp4", ".wav"))
            text_path = os.path.join(directory, filename.replace(".mp4", ".txt"))

            # print(f"Processing: {filename}")

            # Extract audio from the video
            audio_file = extract_audio(video_path, audio_path)
            if not audio_file:
                continue

            # Transcribe the audio
            transcription = transcribe_audio(audio_file, model)

            # Save the transcription to a .txt file
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(transcription)
                print(f"Transcription saved to: {text_path}")

            # Optionally delete the intermediate audio file
            # os.remove(audio_file)
            # print(f"Audio file deleted: {audio_file}")

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe audio from MP4 files using Faster Whisper.")
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing .mp4 files to process."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        help="Faster Whisper model size to use (default: base)."
    )
    return parser.parse_args()

# Run the script
if __name__ == "__main__":
    args = parse_arguments()
    process_videos(args.directory, args.model_size)

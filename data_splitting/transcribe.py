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
        # Replace personal identifiers with placeholders
        text = segment.text
        text = text.replace("Sam Hayne", "[BATSMAN]")
        text = text.replace("delivery", "[BOWLER]")
        # Add more replacements as needed
        transcription += text + " "
    return transcription.strip()

# Main function
def process_audios(input_directory, output_directory, model_size):
    # Load the Faster Whisper model
    print(f"Loading Faster Whisper model ({model_size})...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate through all .mp3 files in the directory
    for filename in tqdm(os.listdir(input_directory)):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(input_directory, filename)
            text_path = os.path.join(output_directory, filename.replace(".mp3", ".txt"))

            # Transcribe the audio
            transcription = transcribe_audio(audio_path, model)

            # Save the transcription to a .txt file
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(transcription)
                print(f"Transcription saved to: {text_path}")

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe audio from MP3 files using Faster Whisper.")
    parser.add_argument(
        "--input_directory",
        type=str,
        default="/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/raw/short_audios",
        help="Directory containing .mp3 files to process."
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/raw/transcriptions",
        help="Directory to save transcriptions."
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
    process_audios(args.input_directory, args.output_directory, args.model_size)

import os
import argparse
from faster_whisper import WhisperModel

# Initialize the Whisper model
model_size = "small.en"
model = WhisperModel(model_size, device="cpu")  # Change to int8 or cpu as needed

def transcribe_audio_files(directory):
    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is an audio file (.mp3 or .wav)
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            audio_path = os.path.join(directory, filename)
            # Transcribe the audio file
            segments, info = model.transcribe(audio_path, beam_size=5)

            # Create a text file with the same base name as the audio file
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(directory, txt_filename)

            # Write the transcription to the text file
            with open(txt_path, "w") as f:
                f.write(f"Detected language '{info.language}' with probability {info.language_probability}\n\n")
                for segment in segments:
                    f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")

            print(f"Transcription saved to {txt_filename}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Transcribe audio files in a directory to text files.")
    parser.add_argument("--directory", type=str, help="Path to the directory containing audio files")

    # Parse the arguments
    args = parser.parse_args()

    # Call the transcription function with the provided directory
    transcribe_audio_files(args.directory)

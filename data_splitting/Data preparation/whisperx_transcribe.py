import os
import whisperx
import json
import gc
import torch


def transcribe_audio_files(input_dir, output_dir, model_size="large-v2", batch_size=16, compute_type="float16"):
    """
    Transcribes all audio files in `input_dir` and saves results in `output_dir`.
    Skips files that already have transcriptions in the output directory.

    Args:
        input_dir (str): Path to the directory containing audio files.
        output_dir (str): Path to the directory where transcriptions will be saved.
        model_size (str): Whisper model size. Defaults to "large-v2".
        batch_size (int): Batch size for transcription. Defaults to 16.
        compute_type (str): Precision type ("float16" or "int8"). Defaults to "float16".
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load WhisperX transcription model
    print("Loading transcription model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    # Loop through each audio file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")

            # Skip transcription if it already exists
            if os.path.exists(output_file):
                print(f"Transcription already exists for {filename}. Skipping.")
                continue

            print(f"Processing file: {filename}...")

            # Transcription
            audio = whisperx.load_audio(file_path)
            result = model.transcribe(audio, batch_size=batch_size)

            # Extract only text and timing (no diarization)
            transcription = [
                {"text": seg["text"], "start": seg["start"], "end": seg["end"]}
                for seg in result["segments"]
            ]

            # Save transcription to JSON
            with open(output_file, "w") as f:
                json.dump(transcription, f, indent=2)
            print(f"Transcription saved to {output_file}")

            # Free GPU memory
            gc.collect()
            torch.cuda.empty_cache()

    print("All files processed.")


def load_transcription(file_path):
    """
    Loads a transcription from a JSON file.

    Args:
        file_path (str): Path to the transcription JSON file.

    Returns:
        list: Transcription segments as a list of dictionaries.
    """
    with open(file_path, "r") as f:
        transcription = json.load(f)
    return transcription


def load_all_transcriptions(output_dir):
    """
    Loads all transcriptions from JSON files in a directory.

    Args:
        output_dir (str): Path to the directory containing transcription JSON files.

    Returns:
        dict: A dictionary where keys are file names and values are transcription segments.
    """
    transcriptions = {}
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(output_dir, filename)
            transcriptions[filename] = load_transcription(file_path)
    return transcriptions


# Example usage
if __name__ == "__main__":
    input_directory = "/home/ACG/data/raw/audios"
    output_directory = "/home/ACG/data/raw/transcripts"

    # Transcribe audio files
    transcribe_audio_files(input_directory, output_directory)

    # # Load a single transcription
    # sample_file = os.path.join(output_directory, "sample.json")
    # if os.path.exists(sample_file):
    #     transcription = load_transcription(sample_file)
    #     print("Sample transcription:")
    #     print(transcription)

    # # Load all transcriptions
    # all_transcriptions = load_all_transcriptions(output_directory)
    # print("All transcriptions loaded.")
    # print(all_transcriptions.keys())

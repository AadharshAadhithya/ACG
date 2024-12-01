import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def mp4_to_mp3(mp4_file, mp3_file):
    try:
        # FFmpeg command to convert MP4 to MP3
        command = [
            "ffmpeg", "-i", mp4_file,  # Input file
            "-vn",                      # No video
            "-acodec", "libmp3lame",    # Use MP3 codec
            "-ar", "44100",             # Audio sampling rate
            "-ac", "2",                 # Stereo audio
            "-ab", "192k",              # Audio bitrate
            mp3_file                    # Output file
        ]
        
        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Converted: {mp4_file} -> {mp3_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {mp4_file}: {e}")

def convert_video(mp4_file_path, mp3_file_path):
    # Check if the MP3 file already exists
    if os.path.exists(mp3_file_path):
        print(f"MP3 file for {os.path.basename(mp4_file_path)} already exists. Skipping conversion.")
    else:
        # Convert MP4 to MP3
        mp4_to_mp3(mp4_file_path, mp3_file_path)

def convert_videos_in_directory(mp4_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List of MP4 files to convert
    mp4_files = [f for f in os.listdir(mp4_dir) if f.endswith(".mp4")]
    
    # Prepare file paths
    file_paths = [(os.path.join(mp4_dir, f), os.path.join(output_dir, f.replace(".mp4", ".mp3"))) for f in mp4_files]

    # Parallel processing using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_video, mp4_file, mp3_file) for mp4_file, mp3_file in file_paths]
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            pass  # This is just to ensure all futures are executed.

if __name__ == "__main__":
    mp4_dir = "../../data/raw/videos"  # Path to the folder containing MP4 files
    output_dir = "../../data/raw/audios"   # Path to the folder where MP3 files should be saved

    convert_videos_in_directory(mp4_dir, output_dir)

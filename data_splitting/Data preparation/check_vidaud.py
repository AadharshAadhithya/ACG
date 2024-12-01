import os

def check_mp3_files(mp4_dir, mp3_dir):
    # Ensure the directories exist
    if not os.path.exists(mp4_dir):
        print(f"Error: The directory {mp4_dir} does not exist.")
        return
    
    if not os.path.exists(mp3_dir):
        print(f"Error: The directory {mp3_dir} does not exist.")
        return

    # Get all MP4 files in the videos directory
    mp4_files = [f for f in os.listdir(mp4_dir) if f.endswith(".mp4")]

    # Check if corresponding MP3 files exist in the audios directory
    missing_files = []

    for mp4_file in mp4_files:
        mp3_file = mp4_file.replace(".mp4", ".mp3")
        mp3_file_path = os.path.join(mp3_dir, mp3_file)
        
        if not os.path.exists(mp3_file_path):
            print(f"Missing MP3 for {mp4_file}")
            missing_files.append(mp4_file)

    if not missing_files:
        print("All MP4 files have corresponding MP3 files.")
    else:
        print(f"\nTotal missing MP3 files: {len(missing_files)}")
        print("Missing files:")
        for missing in missing_files:
            print(missing)

if __name__ == "__main__":
    # Directories for MP4 videos and MP3 audios
    mp4_dir = "../../data/raw/videos"  # Path to the folder containing MP4 files
    mp3_dir = "../../data/raw/audios"  # Path to the folder containing MP3 files

    check_mp3_files(mp4_dir, mp3_dir)

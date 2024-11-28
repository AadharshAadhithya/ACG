import subprocess
import os

# Specify the full path to ffmpeg
ffmpeg_path = r'C:\Program Files\ffmpeg\bin\ffmpeg'

# Paths to the video files to be concatenated
video1_path = r'processed_videos\CricketVideo_part1\scene_37.mp4'
video2_path = r'processed_videos\CricketVideo_part1\scene_39.mp4'
output_path = 'concatenated_video.mp4'

# Ensure the video files exist
if not os.path.exists(video1_path):
    raise FileNotFoundError(f"File not found: {video1_path}")
if not os.path.exists(video2_path):
    raise FileNotFoundError(f"File not found: {video2_path}")

# Step 1: Create a temporary text file with the list of videos
with open('video_list.txt', 'w') as f:
    f.write(f"file '{os.path.abspath(video1_path)}'\n")
    f.write(f"file '{os.path.abspath(video2_path)}'\n")

# Step 2: Use ffmpeg to concatenate the videos
try:
    subprocess.run([
        ffmpeg_path, '-f', 'concat', '-safe', '0', '-i', 'video_list.txt',
        '-c', 'copy', output_path, '-y'
    ], check=True)
    print(f"Videos concatenated successfully into {output_path}")
except FileNotFoundError:
    print(f"Error: ffmpeg not found at {ffmpeg_path}. Ensure the path is correct.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred during concatenation: {e}")
finally:
    # Step 3: Clean up the temporary file
    if os.path.exists('video_list.txt'):
        os.remove('video_list.txt')

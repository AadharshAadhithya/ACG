import cv2
import os
import subprocess

# Input video files
video_paths = ['CricketVideo_part7.mp4']

# Base output directory
base_output_dir = 'processed_videos'

# Ensure base directory exists
os.makedirs(base_output_dir, exist_ok=True)

# Path to ffmpeg
ffmpeg_path = r'C:\Program Files\ffmpeg\bin\ffmpeg'

# Process each video
for video_path in video_paths:
    # Get video name without extension to create a unique folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    final_output_dir = os.path.join(base_output_dir, video_name)
    os.makedirs(final_output_dir, exist_ok=True)

    # Open video for processing
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Variables for scene detection
    prev_hist = None
    scene_start_frame = 0
    scene_frame_count = 0
    scene_times = []

    # Detect scene changes
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and calculate histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Compare histograms to detect scene changes
        if prev_hist is not None:
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if hist_diff < 0.8:  # Adjust threshold as needed
                # Save current scene time
                scene_end_frame = scene_start_frame + scene_frame_count
                scene_times.append((scene_start_frame / fps, scene_end_frame / fps))
                scene_start_frame += scene_frame_count
                scene_frame_count = 0

        prev_hist = hist
        scene_frame_count += 1

    # Append the last scene
    scene_end_frame = scene_start_frame + scene_frame_count
    scene_times.append((scene_start_frame / fps, scene_end_frame / fps))

    cap.release()

    # Split video and save to the output directory
    for scene_index, (start_time, end_time) in enumerate(scene_times):
        output_path = os.path.join(final_output_dir, f'scene_{scene_index + 1}.mp4')

        # Use ffmpeg to extract video and corresponding audio segment
        subprocess.run([
            ffmpeg_path,
            '-i', video_path,                # Input video file
            '-ss', str(start_time),          # Start time
            '-to', str(end_time),            # End time
            '-c:v', 'libx264',               # Re-encode video with H.264 codec
            '-preset', 'fast',               # Encoding speed preset
            '-c:a', 'aac',                   # Re-encode audio
            '-strict', 'experimental',       # For AAC support if required
            output_path,                     # Output file
            '-y'
        ])

    print(f"Processed {video_name} successfully. Results saved in: {final_output_dir}")

print("All videos processed successfully.")

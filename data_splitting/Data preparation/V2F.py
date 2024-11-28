import cv2
import numpy as np
import os

# List of video file paths
video_paths = ['CricketVideo_part2.mp4']  # Add all video paths here


# Base output directory
base_output_dir = 'scene_frames'

if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Process each video
for video_path in video_paths:
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    prev_hist = None
    scene_changes = []
    scene_frames = []
    current_scene = []

    # Create a subdirectory for this video's frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(base_output_dir, video_name)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Save the last scene if it exists
            if current_scene:
                scene_frames.append(current_scene)
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Compare with the previous histogram
        if prev_hist is not None:
            hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

            # Detect scene change based on threshold
            if hist_diff < 0.7:  # Adjust threshold as needed
                scene_frames.append(current_scene)

                # Clear current scene and start a new one
                current_scene = []
                scene_changes.append(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Add the current frame to the current scene
        current_scene.append(frame)
        prev_hist = hist

        # Free memory
        frame = None
        gray = None
        hist = None

    cap.release()

    # Save frames for each scene
    for i, scene in enumerate(scene_frames):
        # Create a directory for each scene
        scene_dir = os.path.join(video_output_dir, f'scene_{i+1}')
        os.makedirs(scene_dir, exist_ok=True)
        for j, frame in enumerate(scene):
            frame_path = os.path.join(scene_dir, f'frame_{j+1}.jpg')
            cv2.imwrite(frame_path, frame)
            frame = None

    print(f"Finished processing video: {video_path}")

print("All videos processed successfully.")

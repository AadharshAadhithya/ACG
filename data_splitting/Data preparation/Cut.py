import subprocess
import os

def split_video(input_video, split_length, output_dir):
    """
    Splits a video into two parts based on the specified split length.

    Args:
        input_video (str): Absolute path to the input video file.
        split_length (float): Length in seconds for the first part (can be fractional).
        output_dir (str): Directory to save the output video parts.

    Returns:
        tuple: Paths to the two output video files (part1_path, part2_path).
    """
    # Specify the full path to ffmpeg
    ffmpeg_path = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Verify input video exists
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")

    # Define output file paths
    video_name = os.path.splitext(os.path.basename(input_video))[0]
    part1_path = os.path.join(output_dir, f"{video_name}_part1.mp4")
    part2_path = os.path.join(output_dir, f"{video_name}_part2.mp4")

    try:
        # Part 1: Extract first 'split_length' seconds (accurate with fractional seconds)
        subprocess.run([
            ffmpeg_path, '-i', input_video, '-t', str(split_length), '-c:v', 'copy', '-c:a', 'copy', part1_path, '-y'
        ], check=True)

        # Part 2: Extract remaining video starting at 'split_length' with re-encoding
        subprocess.run([
            ffmpeg_path, '-ss', str(split_length), '-i', input_video, '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', part2_path, '-y'
        ], check=True)

        print(f"Video split successfully:")
        print(f"Part 1: {part1_path}")
        print(f"Part 2: {part2_path}")

        return part1_path, part2_path

    except subprocess.CalledProcessError as e:
        print(f"Error occurred during video splitting: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Use absolute path for input video
    input_video = r"C:\Users\xl987\PycharmProjects\APLProject\processed_videos\CricketVideo_part1\scene_37.mp4"
    split_length = 4
    # Specify the length (can be fractional) for the first part
    output_dir = r"C:\Users\xl987\PycharmProjects\APLProject\processed_videos\CricketVideo_part1"  # Directory to save the output files

    part1, part2 = split_video(input_video, split_length, output_dir)
    if part1 and part2:
        print(f"First part saved to: {part1}")
        print(f"Second part saved to: {part2}")

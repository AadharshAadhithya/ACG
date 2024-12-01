from pytube.download_helper import download_video
import os

def download_youtube_video_v2(link, output_dir):
    """
    Downloads a YouTube video from the given link and saves it to the output directory using pytube2.
    
    Args:
        link (str): The YouTube video URL.
        output_dir (str): The directory where the video will be saved.
        
    Returns:
        str: The file path of the downloaded video.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading video from: {link}...")
    
    # Download the video
    file_path = download_video(url=link, output_dir=output_dir)
    
    print(f"Download completed: {file_path}")
    return file_path

# Example usage
video_url = "https://www.youtube.com/watch?v=WVCCFgZago8&t=6s"
output_directory = "/home/ACG/data/raw"
download_youtube_video_v2(video_url, output_directory)

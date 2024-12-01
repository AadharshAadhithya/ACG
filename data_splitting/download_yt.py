import os
from pytube import YouTube
from pytube.download_helper import download_video

def download_videos_from_txt(txt_file_path):
    # Ensure the folder exists
    os.makedirs("../data/raw/videos", exist_ok=True)

    # Read the URLs from the .txt file
    with open(txt_file_path, "r") as file:
        urls = file.readlines()

    # Loop through each URL and download the video if not already downloaded
    for url in urls:
        url = url.strip()  # Remove any extra spaces or newline characters
        if not url:
            continue

        # Extract video ID from URL
        video_id = url.split("v=")[-1]
        video_path = f"../data/raw/videos/{video_id}.mp4"

        # Check if the video is already downloaded
        if os.path.exists(video_path):
            print(f"Video {video_id} already exists. Skipping download.")
        else:
            print(f"Downloading video {video_id} from {url}...")
            try:
                # Call the download_video function to download the video
                download_video(url=url, output_path="../data/raw/videos", filename=video_id)
                print(f"Video {video_id} downloaded successfully.")
            except Exception as e:
                print(f"Failed to download video {video_id}: {str(e)}")

if __name__ == "__main__":
    txt_file_path = "../data/videos.txt"  
    download_videos_from_txt(txt_file_path)

mkdir -p audio  # Create the 'audio' directory if it doesn't already exist

for file in *.mp4; do
    ffmpeg -i "$file" -q:a 0 -map a "audio/${file%.mp4}.mp3"
done

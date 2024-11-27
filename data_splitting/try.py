import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


vid_path = "/home/hoffman/Documents/UT/Stuffs/Applied ML/project/data/test.mp4"
from decord import VideoReader
from decord import cpu, gpu

vr = VideoReader(vid_path, ctx=cpu(0))

import matplotlib.pyplot as plt 

frames = vr.get_batch([range(6000,8000)]).asnumpy()




# Assuming frames is a NumPy array of shape (N, 360, 640, 3)
frames = np.array(frames)  # Ensure frames is a NumPy array
pitch_frame = frames[950]  # Reference frame (frame[0])

# Compute absolute differences and count non-zero pixels
diff = np.abs(frames - pitch_frame)
mask = (diff > 50).astype(np.uint8)
nonzero = np.count_nonzero(mask, axis=(1, 2, 3))

# Prepare x-axis for plotting
x = np.arange(0, len(frames))

# Set up the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Video frame
ax1.set_title("Video Frame")
video_img = ax1.imshow(frames[0])  # Initialize with the first frame
ax1.axis('off')  # Remove axes for video

# Subplot 2: Non-zero pixel count plot
ax2.set_title("Non-Zero Pixel Differences")
ax2.set_xlim(0, len(frames))  # Set x-axis range
ax2.set_ylim(0, nonzero.max() + 100)  # Set y-axis range with padding
scatter_plot, = ax2.plot([], [], 'bo', label="Non-Zero Pixels")  # Initialize plot
ax2.set_xlabel("Frame Index")
ax2.set_ylabel("Non-Zero Pixel Count")
ax2.legend()

# Function to update the animation
def update(frame):
    # Update video frame
    video_img.set_data(frames[frame])  # Update the frame being displayed

    # Update scatter plot
    scatter_plot.set_data(x[:frame+1], nonzero[:frame+1])  # Update plot data
    return video_img, scatter_plot

# Create the animation
ani = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)

# Display the animation
plt.show()

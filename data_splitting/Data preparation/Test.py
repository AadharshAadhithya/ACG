import cv2
import numpy as np
import pytesseract
import time
import re
from collections import defaultdict

#  locate Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# count time
start_time = time.time()

# video path
video_path = 'CricketVideo.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = 1920
frame_height = 1080
frame_count = 0
processed_frames = 0
scoreboard_data = []
segment_start_frame = 0
last_min_remaining = None
cut_points = []

# define blue color range
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 30 frames gap
    if frame_count % 30 != 0:
        continue

    processed_frames += 1

    # resolution 1920x1080
    frame = cv2.resize(frame, (frame_width, frame_height))


    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue mask
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    blue_area = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Find boundary
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue


        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_width - x)
        h = min(h, frame_height - y)


        scoreboard = frame[y:y + h, x:x + w]

        # Process images
        gray_scoreboard = cv2.cvtColor(scoreboard, cv2.COLOR_BGR2GRAY)
        _, binary_scoreboard = cv2.threshold(gray_scoreboard, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary_scoreboard = cv2.medianBlur(binary_scoreboard, 3)

        # OCR read text
        text = pytesseract.image_to_string(binary_scoreboard, config='--psm 6 -l eng')  # 指定 PSM 模式和语言

        # clean OCR text
        cleaned_text = re.sub(r'[^\w\s.,:;!?]', '', text).strip()  # 去掉非字母数字符号
        if cleaned_text:
            scoreboard_data.append((frame_count, cleaned_text))

            # obtain "Min Remaining Today" value
            match = re.search(r"Min Remaining Today\s+(\d+\.?\d*)", cleaned_text)
            if match:
                current_min_remaining = float(match.group(1))

                # Check if the "Min Remaining Today" changed
                if last_min_remaining is not None and current_min_remaining != last_min_remaining:
                    cut_points.append((segment_start_frame, frame_count - 1))
                    segment_start_frame = frame_count

                last_min_remaining = current_min_remaining

            print(f"Frame {frame_count}: {cleaned_text}")

        # Visulization
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Scoreboard", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show Frames
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cut_points.append((segment_start_frame, frame_count))

cap.release()
cv2.destroyAllWindows()

# Save score board info
consolidated_data = defaultdict(list)

# Combine the data
for frame_no, text in scoreboard_data:
    consolidated_data[frame_no].append(text)

with open('scoreboard_data.txt', 'w', encoding='utf-8') as f:
    for frame_no, texts in consolidated_data.items():
        combined_text = " ".join(texts)
        f.write(f"Frame {frame_no}: {combined_text}\n")

# Save into file
with open('video_cut_points.txt', 'w') as f:
    for start, end in cut_points:
        f.write(f"Segment: {start} to {end}\n")

# Cut and save the video
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i, (start_frame, end_frame) in enumerate(cut_points):
    output_path = f'video_segment_{i + 1}.mp4'
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_no in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()

cap.release()

# Print run time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Processed {processed_frames} frames out of {frame_count} total frames.")
print("Scoreboard data saved to 'scoreboard_data.txt'")
print("Cut points saved to 'video_cut_points.txt'")
print(f"Total execution time: {elapsed_time:.2f} seconds")

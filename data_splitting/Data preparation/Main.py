# Author: Xin Lyu
# Date: 11/7/2024
# AI cricket matches commentary generator

from moviepy.video.io.VideoFileClip import VideoFileClip
# 加载视频文件
video = VideoFileClip("CricketVideo.mp4")
# 设置每个片段的长度为 5 分钟（单位：秒）
segment_duration = 300
# 获取视频总时长
video_duration = int(video.duration)
# 分割视频
for i in range(0, video_duration, segment_duration):
    # 定义每个片段的开始和结束时间
    start_time = i
    end_time = min(i + segment_duration, video_duration)

    # 截取片段并保存
    video_segment = video.subclip(start_time, end_time)
    output_filename = f"CricketVideo_part{i // segment_duration + 1}.mp4"
    video_segment.write_videofile(output_filename, codec="libx264")

# 关闭视频文件以释放资源
video.close()



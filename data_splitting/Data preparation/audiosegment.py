import os
import subprocess
import librosa
import noisereduce as nr
import numpy as np


def extract_audio(video_path, audio_path, ffmpeg_path):
    """
    从视频中提取音频。
    """
    subprocess.run([
        ffmpeg_path, "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
    ], check=True)


def apply_noise_reduction(audio, sr):
    """
    对音频进行降噪处理，并避免除以零。
    """
    audio = np.where(audio == 0, 1e-10, audio)
    return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.5)


def detect_speech_segments(audio_path, silence_threshold=-30, min_duration=1.0, gap_threshold=0.5):
    """


    Args:
        audio_path (str): 音频文件路径。
        silence_threshold (int): 静音阈值 (dB)。
        min_duration (float): 最短语音段时长（秒）。
        gap_threshold (float): 合并片段的最大间隔时长（秒）。

    Returns:
        list of tuple:  (start, end) 。
    """
    y, sr = librosa.load(audio_path, sr=None)

    # Step 1: 降噪处理
    y_denoised = apply_noise_reduction(y, sr)

    # Step 2: 检测语音片段
    intervals = librosa.effects.split(y_denoised, top_db=abs(silence_threshold))

    # 转换为 (start, end) 时间列表
    segments = [(start / sr, end / sr) for start, end in intervals]

    # Step 3: 过滤过短片段
    segments = [(start, end) for start, end in segments if end - start >= min_duration]

    # Step 4: 合并相邻片段
    merged_segments = merge_close_segments(segments, gap_threshold=gap_threshold)

    # Step 5: 延长语音段结束时间
    audio_duration = librosa.get_duration(y=y, sr=sr)
    adjusted_segments = extend_segments(merged_segments, extension=0.5, audio_duration=audio_duration)

    return adjusted_segments


def merge_close_segments(segments, gap_threshold=0.5):

    if not segments:
        return []

    merged_segments = [segments[0]]
    for current_start, current_end in segments[1:]:
        prev_start, prev_end = merged_segments[-1]
        if current_start - prev_end <= gap_threshold:
            merged_segments[-1] = (prev_start, current_end)
        else:
            merged_segments.append((current_start, current_end))
    return merged_segments


def extend_segments(segments, extension=1, audio_duration=None):

    adjusted_segments = []
    for start, end in segments:
        if audio_duration:
            end = min(end + extension, audio_duration)
        else:
            end += extension
        adjusted_segments.append((start, end))
    return adjusted_segments


def split_video(video_path, segments, output_dir, ffmpeg_path):
    """
    根据语音段切割视频。

    Args:
        video_path (str): 输入视频路径。
        segments (list of tuple): (start, end) 分段时间列表。
        output_dir (str): 输出视频的存储目录。
        ffmpeg_path (str): ffmpeg 可执行文件路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (start_time, end_time) in enumerate(segments):
        output_file = os.path.join(output_dir, f"segment_{i + 1}.mp4")
        subprocess.run([
            ffmpeg_path, "-i", video_path, "-ss", str(start_time), "-to", str(end_time),
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", output_file, "-y"
        ], check=True)
        print(f"保存分段 {i + 1}: {output_file}")


if __name__ == "__main__":
    # 配置路径
    ffmpeg_path = r"C:\Program Files\ffmpeg\bin\ffmpeg"
    video_path = "CricketVideo.mp4"
    audio_path = "cricket_audio.wav"
    output_dir = "processed_videos"

    # Step 1: 提取音频
    print("正在提取音频...")
    extract_audio(video_path, audio_path, ffmpeg_path)

    # Step 2: 检测语音段
    print("正在检测语音分段...")
    speech_segments = detect_speech_segments(audio_path, silence_threshold=-30, min_duration=1.0, gap_threshold=1.0)
    print(f"检测到的语音分段: {speech_segments}")

    # Step 3: 根据语音段切割视频
    print("正在切割视频...")
    split_video(video_path, speech_segments, output_dir, ffmpeg_path)

    print(f"视频处理完成，分段已保存到 {output_dir}")

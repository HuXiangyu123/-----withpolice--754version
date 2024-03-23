import cv2
import subprocess
from moviepy.editor import VideoFileClip
import os
import glob
import librosa

class VideoProcessor:
    def __init__(self):
        self.last_processed_file =None
        self.processed_files = set()  # 存储已处理文件的路径

        self.batch_size = 5  # 每批处理的视频文件数量
        self.processed_count = 0  # 已处理的视频文件计数

    def is_silent(self, file_path, threshold=0.01):
        """Check if the audio file is silent."""
        y, sr = librosa.load(file_path, sr=16000)
        return y.size == 0 or max(abs(y)) < threshold

    def extract_audio_with_moviepy(self, input_video, output_audio='processed_audio.wav'):
        video = VideoFileClip(input_video)
        audio_clip = video.audio.subclip(0, 25)
        audio_clip.write_audiofile(output_audio, fps=16000)
        return output_audio

    def merge_audio(self, video_file, audio_file, output_file):
        command = ['ffmpeg', '-y', '-i', video_file, '-i', audio_file,
                   '-shortest', '-c:v', 'copy', '-c:a', 'aac', '-ar', '16000',
                   '-strict', 'experimental', output_file]
        subprocess.run(command)

    def process_video(self, input_file, temp_video_file):
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            print(f"Error opening video file {input_file}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_file, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

    def load_processed_files(self, file_path):
        """从文本文件中加载已处理文件的路径"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    self.processed_files.add(line.strip())

    def save_processed_files(self, file_path):
        """将已处理文件的路径保存到文本文件中"""
        with open(file_path, 'w') as f:
            for file_path in self.processed_files:
                f.write(file_path + '\n')

    def update_processed_files(self, processed_file_path):
        """更新已处理文件的文本文件"""
        if self.processed_count % self.batch_size == 0:
            self.save_processed_files(processed_file_path)

    def batch_process_folder(self, input_folder, output_folder, processed_file_path):
        # 加载已处理文件的路径
        self.load_processed_files(processed_file_path)

        # 创建输出文件夹，如果不存在的话
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历输入文件夹中的所有MP4文件
        for video_file in glob.glob(os.path.join(input_folder, '*.mp4')):
            base_name = os.path.basename(video_file)
            temp_video = os.path.join(output_folder, f"temp_{base_name}")
            output_video = os.path.join(output_folder, f"processed_{base_name}")

            # 检查是否需要跳过该视频
            if video_file in self.processed_files:
                print(f"Skipping {video_file} as it was already processed.")
                continue

            # 静音检测
            audio_file = self.extract_audio_with_moviepy(video_file)
            if self.is_silent(audio_file):
                silent_folder = os.path.join(output_folder, "silent_videos")
                os.makedirs(silent_folder, exist_ok=True)
                os.rename(video_file, os.path.join(silent_folder, base_name))
                print(f"Video {video_file} is silent. Moved to silent_videos folder.")
                continue

            # 处理视频
            self.process_video(video_file, temp_video)
            self.merge_audio(temp_video, audio_file, output_video)

            # 删除临时音频和视频文件
            os.remove(audio_file)
            os.remove(temp_video)

            print(f"Video processed and saved with audio as {output_video}")

            # 将处理过的文件路径添加到已处理文件集合中
            self.processed_files.add(video_file)
            self.processed_count += 1

            # 更新已处理文件的文本文件
            self.update_processed_files(processed_file_path)

        # 保存最终的已处理文件的路径
        self.save_processed_files(processed_file_path)

if __name__ == "__main__":
    video_processor = VideoProcessor()
    input_folder = "dataset/带时间视频/摄像头三"
    output_folder = "dataset/带时间视频_修复/摄像头三"
    processed_file_path = "dataset/带时间视频/processed_files.txt"  # 已处理文件的路径
    video_processor.batch_process_folder(input_folder, output_folder, processed_file_path)

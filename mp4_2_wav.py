import os
from moviepy.editor import *
import glob

def mp4_2_wavfun(input_folder_path = 'dataset/25s',output_folder_path = 'dataset/25s_audio'):
    # 检查输出文件夹是否存在，如果不存在，则创建
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 获取文件夹中所有MP4文件的列表
    video_files = glob.glob(os.path.join(input_folder_path, '*.mp4'))

    # 遍历所有视频文件并转换
    for video_file in video_files:
        # 获取不带扩展名的文件名，以便用于输出文件
        file_name = os.path.basename(video_file).split('.')[0]
        output_audio_path = os.path.join(output_folder_path, f'{file_name}.wav')

        # 加载视频文件
        video = VideoFileClip(video_file)

        # 截取前25秒
        audiofile = video.audio.subclip(0, 25)
        # 将视频的音频部分导出为WAV文件
        audiofile.subclip(0, 25).write_audiofile(output_audio_path)

       # print(f"转换完成，音频已保存为 {output_audio_path}")

    print("所有视频已成功转换。")

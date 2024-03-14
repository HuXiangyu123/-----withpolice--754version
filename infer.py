import argparse
import functools
import time
import numpy
import pydub

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments
from mp4_2_wav import mp4_2_wavfun
import librosa
import soundfile as sf
import os
import re
import csv

import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

#配置设置
modelname1='CAMPPlus_MFCC'
modelconfig1='cam++.yml'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/'+modelconfig1,   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('model_path',       str,    'models/'+modelname1+'/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
predictor1 = MAClsPredictor(configs=args.configs,
                            model_path=args.model_path,
                            use_gpu=args.use_gpu)


modelname2='CAMPPlus_MelSpectrogram'
modelconfig2='cam++_melspectrogram.yml'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/'+modelconfig2,   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('model_path',       str,    'models/'+modelname2+'/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
predictor2 = MAClsPredictor(configs=args.configs,
                            model_path=args.model_path,
                            use_gpu=args.use_gpu)


modelname3='EcapaTdnn_MFCC'
modelconfig3='ecapa_tdnn_mfcc.yml'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/'+modelconfig3,   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('model_path',       str,    'models/'+modelname3+'/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
predictor3 = MAClsPredictor(configs=args.configs,
                            model_path=args.model_path,
                            use_gpu=args.use_gpu)
#切割音频


def clear_tempslice(file_savepath="dataset/tempslice"):
    if not os.path.exists(file_savepath) or not os.path.isdir(file_savepath):
        print("The specified path does not exist or is not a directory.")
        return
        # 遍历文件夹中的所有文件和文件夹
    for filename in os.listdir(file_savepath):
        file_path = os.path.join(file_savepath, filename)
        try:
            # 如果是文件，则删除
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # 如果需要，也可以添加删除子文件夹的代码
            # elif os.path.isdir(file_path):
            #     shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def extract_number(filename):
    match = re.search(r'(\d+)\.wav$', filename)
    return int(match.group(1)) if match else 0





class Predict:
    def __init__(self, file_path, file_folder,print_sit = 1,csv_result = 'dataset',mode ='simple'):
        self.file_path = file_path
        self.file_folder = file_folder
        self.print_sit = print_sit
        self.csv_result =csv_result
        self.mode = mode
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)

    def is_silent(self, threshold=0.01):
        """Check if the audio file is silent."""
        y, sr = librosa.load(self.file_path, sr=16000)

        return y.size == 0  or max(abs(y)) < threshold
    def slice_audio(self, slice_length_sec=5 ,slide_step_sec=5.0, samplerate=16000,
                    file_savepath="dataset/tempslice"):
        # 加载音频文件，y是音频信号，sr是采样率
        divide_true = 0
        file_path=self.file_path
       # print(file_path)
        y, sr = librosa.load(file_path, sr=samplerate)

        # 计算音频长度（秒）
        audio_length_sec = librosa.get_duration(y=y, sr=sr)
        #print(audio_length_sec)
        divide_true = audio_length_sec > 5.5 #避免5.2 5.1 的情况也切片
        # 如果音频长度大于5.5秒，则开始切片
        if divide_true:

            slice_length_samples = int(slice_length_sec * sr)
            slide_step_samples = int(slide_step_sec * sr)

            start_sample = 0
            slice_index = 1

            while start_sample + slice_length_samples <= len(y):
                end_sample = start_sample + slice_length_samples

                # 提取音频片段
                slice_y = y[start_sample:end_sample]

                # 导出音频片段

                slice_file_name = f"slice_{slice_index}.wav"
                sf.write(file_savepath + '/' + slice_file_name, slice_y, sr)
                start_sample += sr
                slice_index += 1

        return divide_true

    def predictset_5saudio(self):


        label_pre1,_= predictor1.predict(audio_data=self.file_path )
        label_pre2,_ = predictor2.predict(audio_data=self.file_path)
        label_pre3,_ = predictor3.predict(audio_data=self.file_path)
        votes_for_1 = [label_pre1, label_pre2, label_pre3].count('1')

        # 如果标签 '1' 的票数超过半数，则最终标签为 '1'，否则为 '0'
        label_pre = '1' if votes_for_1 > 1 else '0'
        if  self.print_sit == 1:
            result = "炸街" if label_pre == '1' else "不是炸街"
            print(f"{self.file_path}预测结果为{result}")
        return label_pre

    def predictset_longaudio(self, file_slicepath="dataset/tempslice"):
        wavfiles = [os.path.join(root, file)
                    for root, dirs, files in os.walk(file_slicepath)
                    for file in files if file.endswith('.wav')]
        wavfiles = sorted(wavfiles, key=extract_number)

        labels = []
        for index, audio_path in enumerate(wavfiles):
            label_pre1, _ = predictor1.predict(audio_data=audio_path)
            label_pre2, _ = predictor2.predict(audio_data=audio_path)
            label_pre3, _ = predictor3.predict(audio_data=audio_path)
            votes_for_1 = [label_pre1, label_pre2, label_pre3].count('1')

            # 如果标签 '1' 的票数超过半数，则最终标签为 '1'，否则为 '0'
            label_pre = '1' if votes_for_1 > 1 else '0'
            labels.append(label_pre)
            if self.print_sit == 1 and index % 10 == 0:
                print(f'单音频切片预测进度：{index}/{len(wavfiles)}')
        # 使用滑动窗口检查连续性
        window_size = 5
        result = 0
        consecutive_indices = [i for i in range(len(labels) - window_size + 1)
                               if labels[i:i + window_size] == ['1'] * window_size]

        if consecutive_indices:
            result = 1
            if self.print_sit == 1:
                for index, start_index in enumerate(consecutive_indices):
                    print(f"在{start_index + 1}秒处，发生第{index + 1}次炸街现象")
        else:
            if self.print_sit == 1:
                print("未发生炸街现象")

        return result

        # results_df.to_excel('testslice.xlsx', index=False)


    def predictset(self):


        if self.mode == 'simple':
            self.print_sit = 1
        else:
            self.print_sit = 0

        if self.is_silent():
            return '0'
       # print(self.file_path)
        divide_sit = self.slice_audio()

        if divide_sit == 1:
            result = self.predictset_longaudio()
        else:
            result = self.predictset_5saudio()
        return result

    #连续预测
    def predictset_list(self):
        self.mode = 'list'
        listfolder = os.listdir(self.file_folder)
        csv_file_path = os.path.join(self.csv_result,time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + 'Prediction_Results.csv')

        # 打开CSV文件准备写入
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:

            csv_writer = csv.writer(csvfile)
            # 写入CSV文件的标题行，根据你的预测结果自定义列名
            csv_writer.writerow(['File Name', 'Prediction Result'])

            for index, file_name in enumerate(listfolder):
                file_path = os.path.join(self.file_folder, file_name)
                self.file_path = file_path
                # 假设predictset返回预测结果
                self.print_sit = 0
                prediction_result = self.predictset()

                # 将文件名和预测结果写入CSV文件
                csv_writer.writerow([file_name, prediction_result])

                if index % 10 == 0:
                    print(f'预测进度：{index}/{len(listfolder)}')
        self.mode = 'simple'
# add_arg('audio_path', str, audiopath, '音频路径')

# args = parser.parse_args()
# print_arguments(args=args)
# predictor = MAClsPredictor(configs=args.configs,
#                            model_path=args.model_path,
#                            use_gpu=args.use_gpu)


# label, score = predictor.predict(audio_data=args.audio_path)

# print(f'音频：{args.audio_path} 的预测结果标签为：{label}，得分：{score}')

if __name__ ==  "__main__":
    #把25s文件夹中的视频 音频提取到25s audio（提取一次之后将mp4_2_wavfun注释即可，避免重复转换）

    #librosa可以直接处理视频，所以可以直接用视频输入预测，MP4-2-wav不出bug可以不适用
    #mp4_2_wavfun(input_folder_path = 'dataset/25s/videos_total/1区')
    pred=Predict(file_path='dataset/25s/videos_total/1区/48B02DE04B0A_1704394212_1704394217_kIigJ.mp4',file_folder='dataset/25s/videos_total/4区/')
    pred.predictset_list()
    #pred.predictset()
    clear_tempslice()
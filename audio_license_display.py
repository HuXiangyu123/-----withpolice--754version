from abc import ABC

import os
import io
import av
import cv2
import librosa
import subprocess
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
import time


# 绘制音轨图
class drawLineToVideo(ABC):
    def __init__(self, width, height, audioDataFrameList, fps) -> None:
        super().__init__()
        self.fps = int(fps)
        self.width = width
        self.height = height
        self.audioDataFrameList = audioDataFrameList

    def saveAudioLineVideo(self, outPath, roaringIndex):
        # 创建一个视频写入器对象，准备将图像帧写入视频文件

        output_directory = os.path.dirname(outPath)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        frameIndex = 0
        audioDataFrameList = np.array(self.audioDataFrameList)
        fourcc = cv2.VideoWriter_fourcc(*'FLV1')  # FourCC编码器，以FLV1作为视频编码格式
        out = cv2.VideoWriter(outPath, fourcc, self.fps, (self.width, self.height))

        frame_length = len(audioDataFrameList[0])  # 每帧中采样点数
        audio_length = len(audioDataFrameList.flatten())  # 整个音频采样点数
        print(roaringIndex)
        roaringIndex = [num // 2 * frame_length for num in roaringIndex]  # 因计算阈值时，短时能量存在hop_length，列表标号为实际帧的两倍，故除以2.
        print(roaringIndex)
        # 采用布尔数组，对炸街音频中的采样点进行标记
        marked = [False] * audio_length
        for start, end in np.array(roaringIndex).reshape(-1, 2):
            for i in range(start, end + 1):
                marked[i] = True

        while True:
            # 取数据
            # 检查剩余的音频数据帧是否足够生成一个完整的视频帧 
            if (len(self.audioDataFrameList) - frameIndex) > 2 * self.fps:
                display_range = 2 * self.fps
            else:
                display_range = len(self.audioDataFrameList) - frameIndex
            # 对每一帧都生成一个长达64ms(16000/15.625)*25(视频的fps)的音轨图
            audio_data = audioDataFrameList[frameIndex:frameIndex + display_range, :].flatten()

            # 视频fps：25 音频fps：15.625
            # 创建折线图
            image = np.zeros((self.height, self.width, 3), np.uint8)
            step = self.width / len(audio_data)  # 每个数据点之间的水平间隔
            # 将一帧图像的全部数据绘制成折线图
            for i in range(len(audio_data) - 1):
                x1 = int(i * step)
                y1 = int(audio_data[i] * self.height / 2 * 1.5 + self.height / 2)
                x2 = int((i + 1) * step)
                y2 = int(audio_data[i + 1] * self.height / 2 * 1.5 + self.height / 2)
                # 对数据范围内的折线图进行绘制，其中炸街部分红色，非炸街部分为白色
                if (i + frame_length * frameIndex) < audio_length:
                    # print(i+frame_length*frameIndex,frameIndex)
                    if marked[i + frame_length * frameIndex]:
                        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    else:
                        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            out.write(image)
            # 判断数据是否取完
            frameIndex = frameIndex + 1
            if frameIndex == len(audioDataFrameList):
                break

            print("drawLineToVideo:", (frameIndex / len(audioDataFrameList)) * 100 , end="%\r")

        out.release()


class drawSTFTToVideo(ABC):
    def __init__(self, width, height, audioDataFrameList, fps, sr) -> None:
        super().__init__()
        self.fps = int(fps)
        self.width = width
        self.height = height
        self.audioDataFrameList = audioDataFrameList
        self.sr = sr

    def saveAudioSTFTToVideo(self, outPath, sr, ):
        frameIndex = 0
        audioDataFrameList = np.array(self.audioDataFrameList)

        fourcc = cv2.VideoWriter_fourcc(*'FLV1')
        out = cv2.VideoWriter(outPath, fourcc, self.fps, (self.width, self.height))

        while True:
            # 取数据
            if (len(self.audioDataFrameList) - frameIndex) > self.fps:
                fps = self.fps
            else:
                fps = len(self.audioDataFrameList) - frameIndex
            # 对每一帧都生成一个长达64ms(16000/15.625)*25(视频的fps)的stft图
            audio_data = audioDataFrameList[frameIndex:frameIndex + fps, :].flatten()

            frameIndex = frameIndex + 1
            # 计算短时傅里叶变换（STFT）
            frame_length = int(len(audio_data) / 5)
            hop_length = int(frame_length / 2)
            stft = librosa.stft(audio_data, n_fft=frame_length,
                                win_length=frame_length, hop_length=hop_length)
            # stft = librosa.stft(audio_data, n_fft=len(audioDataFrameList[0]))

            # 将STFT转换为分贝单位
            stft_db = librosa.amplitude_to_db(np.abs(stft))

            # 绘制频谱图
            # 将图像转换为 numpy 数组
            librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log',
                                     vmin=-40, vmax=40, cmap='coolwarm')
            # librosa.display.specshow(stft_db, x_axis='time', y_axis='log',vmin=-60, vmax=60,cmap='coolwarm')
            plt.title('Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            buf = io.BytesIO()  # 字节缓冲区，用于保存绘制的频谱图
            plt.savefig(buf, format='png')  # 将绘制的频谱图保存到字节缓冲区中
            plt.clf()  # 清除当前的图形。label='Amplitude'
            buf.seek(0)  # 将字节缓冲区的读取位置设置为起始位置。
            img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), 1)
            # 将图像从 RGB 格式转换为 BGR 格式
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (self.width, self.height))
            out.write(image)

            # 判断数据是否取完
            if frameIndex == len(audioDataFrameList):
                break

            print("drawSTFTToVideo:", (frameIndex / len(audioDataFrameList)) * 100, end="%\r")

        out.release()


class drawRMSToVideo(ABC):
    def __init__(self, width, height, audioDataFrameList, fps) -> None:
        super().__init__()
        self.fps = int(fps)
        self.width = width
        self.height = height
        self.audioDataFrameList = audioDataFrameList

    def saveAudioRMSToVideo(self, outPath, ):
        frameIndex = 0
        audioDataFrameList = np.array(self.audioDataFrameList)
        fourcc = cv2.VideoWriter_fourcc(*'FLV1')
        out = cv2.VideoWriter(outPath, fourcc, self.fps, (self.width, self.height))

        while True:
            # 取数据
            if (len(self.audioDataFrameList) - frameIndex) > 2 * self.fps:
                display_range = 2 * self.fps
            else:
                display_range = len(self.audioDataFrameList) - frameIndex

            audio_data = np.array(self.audioDataFrameList[frameIndex:frameIndex + display_range]).flatten()  # 1s音频数据
            frameIndex = frameIndex + 1

            # 计算短时能量
            energy = librosa.feature.rms(y=audio_data, frame_length=len(audioDataFrameList[frameIndex]))[0]

            # 绘制短时能量图
            xList = np.arange(0, len(energy))
            plt.plot(xList, energy)
            plt.title(f'Short-term Energy')
            plt.xlabel('frame')
            plt.ylim(0, 0.15)
            plt.ylabel('Energy')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.clf()
            buf.seek(0)
            img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), 1)
            # 将图像从 RGB 格式转换为 BGR 格式
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (self.width, self.height))
            out.write(image)

            # 判断数据是否取完
            if frameIndex == len(audioDataFrameList) - 1:
                break

            print("drawRMSToVideo:", (frameIndex / len(audioDataFrameList)) * 100, end="%\r")

        out.release()


class audio(ABC):
    def __init__(self, videoPath) -> None:
        super().__init__()
        self.videoPath = videoPath
        self.audioDataFrameList = []
        self.roaringIndex = []

    def audioPreProcess(self, audioDataFrameList, sr, frame_num, frame_length):
        waveform = np.array(audioDataFrameList).flatten()
        # 音频降噪去白噪声
        win_length = 1024  # 帧长度为20ms 或1024
        hop_length = 512  # 帧移为10ms 或512
        S_noisy = librosa.stft(waveform, n_fft=1024, hop_length=hop_length, win_length=win_length)
        D, T = np.shape(S_noisy)
        Mag_noisy = np.abs(S_noisy)  # 幅度
        Phase_nosiy = np.angle(S_noisy)  # 相位
        Power_nosiy = Mag_noisy ** 2  # 得到信号的能量谱
        # 估计噪声信号的能量
        # 由于噪声信号未知 这里假设 含噪（noisy）信号的前30帧为噪声
        # Mag_nosie = np.mean(np.abs(S_noisy[:, :30]), axis=1, keepdims=True)  # 沿T的维度取均值，输出维度是129*1
        # Power_nosie = Mag_nosie ** 2
        Power_nosie = np.load('noise_energy.npy')
        Power_nosie = np.tile(Power_nosie, [1, T])  # 对前30帧进行不断复制到与带噪语音等长
        # 幅度减
        Mag_enhenc = np.sqrt(Power_nosiy) - np.sqrt(Power_nosie)
        Mag_enhenc[Mag_enhenc < 0] = 0

        # 对信号进行恢复 降噪后信号为y
        S_enhec = Mag_enhenc * np.exp(1j * Phase_nosiy)
        y = librosa.istft(S_enhec, hop_length=hop_length, win_length=win_length)
        # 带通滤波器滤波
        # ————————————————————————————————————————————————————
        low_freq = 50
        high_freq = 800
        low = low_freq / (sr / 2)
        high = high_freq / (sr / 2)
        order = 4
        # 使用巴特沃斯带通滤波器设计滤波器系数
        b, a = signal.butter(order, [low, high], btype='band')
        # 应用滤波器进行降噪
        filtered_waveform = signal.lfilter(b, a, y)
        step_size = win_length // 2  # 步长大小

        # 对降噪后能量过小部分置为零
        window_size = frame_length  # 窗口大小
        threshold = 0.004  # 能量阈值
        for i in range(len(filtered_waveform)):
            start = max(0, i - window_size // 2)
            end = min(len(filtered_waveform), i + window_size // 2)
            energy = np.sum(filtered_waveform[start:end] ** 2)
            if energy < threshold:
                filtered_waveform[i] = 0
        sf.write("OUT.wav", filtered_waveform, sr)
        filtered_waveform = filtered_waveform.reshape(frame_num, frame_length)
        audioDataFrameList = filtered_waveform.tolist()
        # print('audio:',len(audioDataFrameList))
        # print('audio:',len(audioDataFrameList[-1])) 确认和packet解包之后维度相同
        return audioDataFrameList

    def detectRoaringAudio(self, audioDataFrameList, sr, frame_length):
        waveform = np.array(audioDataFrameList).flatten()
        energy = librosa.feature.rms(y=waveform, frame_length=frame_length)[0]
        # print('frame_length:',frame_length)
        # print('len(energy):',len(energy))
        MH = 0.04
        ML = 0.005
        A = []
        B = []
        result = []

        # 首先利用较大能量阈值 MH 进行初步检测
        flag = 0
        for i in range(len(energy)):
            if len(A) == 0 and flag == 0 and energy[i] > MH:
                A.append(i)
                flag = 1
            elif flag == 0 and energy[i] > MH and i - 21 > A[len(A) - 1]:
                A.append(i)
                flag = 1
            elif flag == 0 and energy[i] > MH and i - 21 <= A[len(A) - 1]:
                A = A[:len(A) - 1]
                flag = 1  # 将上一个语音开始点删除

            if flag == 1 and energy[i] < MH:
                A.append(i)
                flag = 0
        print("较高能量阈值A:" + str(A))

        # 利用较小能量阈值 ML 进行第二步能量检测
        for j in range(len(A)):
            i = A[j]
            if j % 2 == 1:
                while i < len(energy) and energy[i] >= ML:
                    i = i + 1
                B.append(i)
            else:
                while i > 0 and energy[i] >= ML:
                    i = i - 1
                B.append(i)

        if len(B) % 2 == 1:
            # B.pop()
            B.append(len(energy))
        print("较低能量阈值B:" + str(B))

        #     数据处理,对重叠数据删除
        i = 0
        while i < len(B) - 3:
            start1, end1, start2, end2 = B[i], B[i + 1], B[i + 2], B[i + 3]

            # 检查后一组起点是否小于前一组终点
            if start2 <= end1:
                # 删除重叠数据
                print(B[i + 1:i + 3])
                del B[i + 1:i + 3]  # 含头不含尾
            else:
                # 添加非重叠数据到结果列表
                result.extend([start1, end1])
                i += 2
        # 添加最后一组非重叠数据
        result.extend(B[-2:])
        print('result:', result)
        return result

    def extractAudioVideo(self, ):
        container = av.open(self.videoPath)  # av.open打开音频文件
        # 寻找到音频流
        audio_stream = None
        for stream in container.streams:
            if stream.type == 'audio':
                audio_stream = stream
                break

        # 没找到
        if audio_stream is None:
            return

        # 1帧64ms，包含1024个数据，每秒15.625帧
        for packet in container.demux(audio_stream):
            # 将给定的一组流（Streams）进行解复用（demux），也就是将多路复用的数据流拆分成独立的数据包（Packet）,通常是解码
            try:
                for frame in packet.decode():
                    # 对解码后的帧进行处理
                    if isinstance(frame, av.AudioFrame):
                        # 检查帧是否为音频帧，并将音频数据添加到 self.audioDataFrameList 列表中
                        audio_data = frame.to_ndarray()  # audio_data(1,1024);audio_data[0](1024,)
                        self.audioDataFrameList.append(
                            audio_data[0])  # audio_data[0]为数组，self.audioDataFrameList为元素为array的列表
            except av.error.EOFError:
                break

        sr = audio_stream.rate
        frame_num = len(self.audioDataFrameList)  # 帧数
        frame_length = len(audio_data[0])  # 每帧长度
        # ——————————————————————————————————————————————————————
        self.audioDataFrameList = self.audioPreProcess(self.audioDataFrameList, sr, frame_num, frame_length)
        self.roaringIndex = self.detectRoaringAudio(self.audioDataFrameList, sr, frame_length)
        # ——————————————————————————————————————————————————————

        # 获取视频的帧率和分辨率
        cap = cv2.VideoCapture(self.videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # 生成音轨视频
        dltv = drawLineToVideo(1920, 270, self.audioDataFrameList, fps)
        dltv.saveAudioLineVideo("temp/dltv.flv", self.roaringIndex)
        # 生成频谱图

        dsv = drawSTFTToVideo(480, 405, self.audioDataFrameList, fps, sr)
        dsv.saveAudioSTFTToVideo("temp/dsv.flv", sr)
        # 生成短时能量图
        dsv = drawRMSToVideo(480, 405, self.audioDataFrameList, fps)
        dsv.saveAudioRMSToVideo("temp/rms.flv")

    def saveAudioVideo(self, outPath, ):
        # 打开第一个视频文件
        cap1 = cv2.VideoCapture(self.videoPath)

        # 打开第二个视频文件（原始波形）
        cap2 = cv2.VideoCapture("temp/dltv.flv")

        # 打开第三个视频文件（频谱）
        cap3 = cv2.VideoCapture("temp/dsv.flv")

        # 打开第四个视频文件(能量)
        cap4 = cv2.VideoCapture("temp/rms.flv")

        # 获取视频1帧率,帧数
        fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
        frame1_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_all_count = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

        # 获取视频2帧率,帧数
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
        frame2_count = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

        # 获取视频3帧率,帧数
        fps3 = int(cap3.get(cv2.CAP_PROP_FPS))
        frame3_count = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))

        # 获取视频3帧率,帧数
        fps4 = int(cap4.get(cv2.CAP_PROP_FPS))
        frame4_count = int(cap4.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'FLV1')
        out = cv2.VideoWriter("OUT.flv", fourcc, fps1, (1920, 1080))

        frame_time_index = frame1_count / frame2_count  # (25/15.625=1.6025641025641026)
        # 遍历视频1的所有帧
        frame1_count = 0
        frame2_count = 0
        roaringIndex = [num // 2 * frame_time_index for num in self.roaringIndex]
        while True:
            ret1, frame1 = cap1.read()
            if not ret1:
                break
            else:
                frame1_count = frame1_count + 1
            frame1 = cv2.resize(frame1, (1440, 810))
            # print(ret1,frame1_count, frame1.shape)
            # 只输出583帧frame1，583*40ms=23.32ms，少了1.68ms，刚好等于音频处理一次数据时长=1
            # 从视频2中读取一帧
            if ((frame2_count * frame_time_index) - frame1_count) < 1:
                ret2, frame2 = cap2.read()
                if not ret2:
                    break
                else:
                    frame2_count = frame2_count + 1

                ret3, frame3 = cap3.read()
                if not ret3:
                    break

                ret4, frame4 = cap4.read()
                if not ret4:
                    break

            # 缩放和裁剪视频1的帧并将其复制到输出图像中
            out_frame = np.zeros((1080, 1920, 3), np.uint8)
            out_frame[:810, :1440] = frame1

            # 缩放和裁剪视频2的帧并将其复制到输出图像中
            out_frame[810:, :] = frame2
            out_frame[:404, 1440:1920] = frame3
            out_frame[406:810, 1440:1920] = frame4

            # 画分割线条,转换后视频中存在炸街的帧框体变红
            # roaringIndex = [num //2 *frame_time_index  for num in self.roaringIndex]
            if any(start <= frame1_count <= end for start, end in zip(roaringIndex[::2], roaringIndex[1::2])):
                # 在空白图像上绘制图片
                # img = cv2.imread(r'F:\360MoveData\Users\qo0v0op\Desktop\Roar\炸街my\车辆.jpg')
                # # img = cv2.imread("F:/360MoveData/Users/qo0v0op/Desktop/Roar/炸街my/车辆.jpg")
                # img_size = (250, 140)
                # img = cv2.resize(img, img_size)
                # img_x = 20
                # img_y = 20
                # out_frame[img_y:img_y + img_size[1], img_x:img_x + img_size[0]] = img

                # 在空白图像上绘制文本
                word = 'license'
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f'vehicle:{word}'
                text_size, _ = cv2.getTextSize(text, font, 1, 2)  # 文字，字体，字体大小，字体粗细
                text_x = 40
                text_y = 190
                # text_y = img_y + img_size[1] + 30
                text_color = (255, 255, 255)

                # 在空白图像上绘制黑色矩形
                rect_x = text_x - 25
                rect_y = text_y - text_size[1]
                rect_w = text_size[0] + 5
                rect_h = text_size[1] + 5
                cv2.rectangle(out_frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 0, 0), -1)

                cv2.putText(out_frame, text, (text_x, text_y), font, 0.8, text_color, 2)
                color = (0, 0, 255) if int(time.time() * 10) % 2 == 0 else (255, 255, 255)

            else:
                color = (0, 255, 0)
            cv2.line(out_frame, (0, 0), (0, 1080), color, 10)
            cv2.line(out_frame, (0, 0), (1920, 0), color, 10)
            cv2.line(out_frame, (0, 1080), (1920, 1080), color, 10)
            cv2.line(out_frame, (1920, 0), (1920, 1080), color, 10)

            cv2.line(out_frame, (0, 810), (1920, 810), color, 5)
            cv2.line(out_frame, (1440, 405), (1920, 405), color, 5)
            cv2.line(out_frame, (1440, 0), (1440, 810), color, 5)
            # 显示合并后的图片
            out.write(out_frame)
            print("saveAudioVideo", (frame1_count / frame_all_count) * 100, end="\r")

        # 释放视频对象
        cap1.release()
        cap2.release()
        out.release()

        # 处理音频

        # 使用 ffmpeg 工具将 1.flv 文件中的音频保存为 1.wav 文件
        # subprocess.call(['ffmpeg', '-i', self.videoPath, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', 'OUT.wav','-y'])

        # 使用 ffmpeg 工具将 1.wav 文件和 2.flv 文件融合为一个输出文件 output.flv
        # 视频音频未对齐！！！！！！！！！！！

        subprocess.call(
            ['ffmpeg', '-i', 'OUT.flv', '-i', 'OUT.wav', '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
             outPath, '-y'])

        # os.remove("OUT.wav")
        os.remove("OUT.flv")


if __name__ == "__main__":
   a = audio(videoPath="dataset/带时间视频_修复/摄像头四/processed_2024.01.18.08.34.mp4")
   a.extractAudioVideo()
   a.saveAudioVideo('test.flv')
#
# if __name__ == "__main__":
#      folder_path = "F:\\炸街视频\\线上视频_20240118\\2a\\48B02DE039CB_1703540001_1703540016_4SNAW.mp4" # 指定路径
#      file_names = os.listdir(folder_path)
#      for file_name in file_names:
#          if file_name.endswith('.flv'):  # 判断文件名是否以 .flv 结尾
#              file_path = os.path.join(folder_path, file_name)  # 拼接文件路径
#              print(file_path)
#              a = audio(file_path)
#              a.extractAudioVideo()
#              output_path="out-"+file_name
#              print(output_path)
#              a.saveAudioVideo(output_path)

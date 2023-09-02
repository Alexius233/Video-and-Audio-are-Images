import torch
import numpy as np
import librosa

# mel-spectrogram parameters
#SR = 22050 # 采样率
#N_FFT = 2048
#HOP_LEN = 512
#DURA = 5.20   # 采样时间
#N_MEL = 224

# hop_len 的作用是 : Sr * DURA / hop_len 影响wide, n_mel 的作用是 : 决定height,n_mel是多少height是多少
# n_fft : 似乎是影响了分贝数,特征的整体数值大小

def Audio_feature_extractionder(indir, is_train=True):
    SR = 22050 # 采样率
    N_FFT = 2048
    HOP_LEN = 512
    DURA = 5.20   # 采样时间
    N_MEL = 224

    # Load audio
    audio_name = indir
    src, sr = librosa.load(indir, sr=SR)  # value num = Sr * DURA ， 原始长度

    # 原始长度
    n_sample = src.shape[0]
    gap = int(n_sample / 9)  #间隔
    # 需要的长度
    segment = int(DURA * SR)    # 114660 这是一个图的所需大小，需要9段
    half_seg = int(segment / 2)

    time_window = 25  # 时间窗长度
    window_length = sr / 1000 * time_window  # 转换过来的视窗长度
    window_nums = int(segment / window_length)  # 视窗个数

    begin = -11025 # 取值偏移0.5s
    for num in range(0,9):
        begin = begin + gap - half_seg
        end = begin + segment

        if begin < 0: # 防止时长不够
            data_seg = src[0:segment]
        else: # 够的话取一段
            data_seg = src[begin:end]

        if end > n_sample:
            ends = n_sample - 1
            start = n_sample - 1 - segment
            data_seg = src[start:ends]

        coeff = 0.97  # 预加重系数
        time_window = 25  # 时间窗长度
        window_length = int(sr / 1000 * time_window)  # 转换过来的视窗长度：400
        frameNum = int(segment / window_length)  # 视窗个数

        frameData = np.zeros((window_length, frameNum))  # 创建一个空的


         # 汉明窗
        hamwin = np.hamming(window_length)

        for i in range(frameNum):
            singleFrame = data_seg[np.arange(i * window_length, min(i * window_length + window_length, segment))]
            singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coeff * singleFrame[1:])  # 预加重
            frameData[:len(singleFrame), i] = singleFrame
            frameData[:, i] = hamwin * frameData[:, i]  # 加窗

        frameData = np.transpose(frameData)
        length = frameData.shape[0] * frameData.shape[1]
        frameData = np.reshape(frameData, length)

        data_seg = np.hstack((frameData, np.zeros(segment - length))) # 要是segment长度不够拿0补上

        
        y_harmonic, y_percussive = librosa.effects.hpss(data_seg) # 这里的谐波可以备用着，效果可能更好
        logam = librosa.amplitude_to_db

        fv_mel = logam(librosa.feature.melspectrogram(y=data_seg, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MEL))  # -> mel
        fv_mel = torch.tensor(fv_mel).unsqueeze(0) # 拓展成[1, 224, 224]
        fv_mel = torch.cat((fv_mel, fv_mel, fv_mel),0).unsqueeze(0)  # [3, 224,224] -> [1, 3, 224, 224]

        if num == 0:
            MelSpectrogram = fv_mel
        else:
            MelSpectrogram = torch.cat((MelSpectrogram, fv_mel),0)  # 应该是[9, 3, 224, 224]

    # 优先抽取log的mel谱
    # hop_len 的作用是 : Sr * DURA / hop_len 影响wide, n_mel 的作用是 : 决定height,n_mel是多少height是多少
    # n_fft : 似乎是影响了分贝数,特征的整体数值大小



    if is_train == True:
        # return MelSpectrogram是tensor
        return MelSpectrogram

    else:
        return MelSpectrogram, audio_name
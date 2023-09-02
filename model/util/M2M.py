import torch
import torch.utils.data
from torchvision import transforms

import os
import numpy as np
import librosa
import cv2
import random
import math
import warnings

from MMAE.util.decoder import utils
from MMAE.util.decoder.random_erasing import RandomErasing
from MMAE.util.decoder.transform import create_random_augment

warnings.filterwarnings("ignore")


def audio_decode(path, start_idx, end_idx, num_samples, ):  # 待定，暂时没找到视频读取音频的方案
    SR = 44100  # 采样率
    N_FFT = 2048
    HOP_LEN = 512
    DURA = 2.60  # 采样时间
    N_MEL = 224

    # mel的宽度: DURA * SR ÷ HOP_LEN
    #      高度: N_MEL

    src, sr = librosa.load(path, sr=SR)  # value num = Sr * DURA ， 原始长度
    # 原始长度
    n_sample = src.shape[0]
    length = 114688
    # 需要的长度
    segment = int(DURA * SR)  # 114660 这是一个图的所需大小，需要9段
    half_seg = int(segment / 2)

    if type(start_idx) == int:
        index = torch.linspace(start_idx, end_idx, num_samples)

        for num in range(0, len(index)):
            left = int(index[num]) - half_seg
            right = left + segment

            if left < 0:
                data_seg = src[0:segment]
            elif right > 0:
                left = n_sample - segment
                data_seg = src[left:n_sample]
            else:
                data_seg = src[left:right]

            data_seg = np.hstack((data_seg, np.zeros(length - segment)))

            logam = librosa.amplitude_to_db
            fv_mel = logam(
                librosa.feature.melspectrogram(y=data_seg, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MEL))  # -> mel
            fv_mel = fv_mel[:, 0:224]
            fv_mel = torch.tensor(fv_mel).unsqueeze(0)  # 拓展成[1, 224, 224]
            fv_mel = torch.cat((fv_mel, fv_mel, fv_mel), 0).unsqueeze(0)  # [3, 224,224] -> [1, 3, 224, 224]

            if num == 0:
                MelSpectrogram = fv_mel
            else:
                MelSpectrogram = torch.cat((MelSpectrogram, fv_mel), 0)  # 应该是[4, 3, 224, 224]

        return MelSpectrogram

    else:

        lens = len(start_idx)
        for clip_idx in range(0, lens):
            index = torch.linspace(start_idx[clip_idx], end_idx[clip_idx], num_samples)

            for num in range(0, len(index)):
                left = int(index[num]) - half_seg
                right = left + segment

                if left < 0:
                    data_seg = src[0:segment]
                elif right > 0:
                    left = n_sample - segment
                    data_seg = src[left:n_sample]
                else:
                    data_seg = src[left:right]

                data_seg = np.hstack((data_seg, np.zeros(length - segment)))

                logam = librosa.amplitude_to_db
                fv_mel = logam(
                    librosa.feature.melspectrogram(y=data_seg, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MEL))  # -> mel
                fv_mel = fv_mel[:, 0:224]
                fv_mel = torch.tensor(fv_mel).unsqueeze(0)  # 拓展成[1, 224, 224]
                fv_mel = torch.cat((fv_mel, fv_mel, fv_mel), 0).unsqueeze(0)  # [3, 224,224] -> [1, 3, 224, 224]

                if num == 0:
                    MelSpectrogram = fv_mel
                else:
                    MelSpectrogram = torch.cat((MelSpectrogram, fv_mel), 0)  # 应该是[9, 3, 224, 224]

            if clip_idx == 0:
                MelSpectrograms = torch.tensor(MelSpectrogram).unsqueeze(0)
            else:
                MelSpectrogram = torch.tensor(MelSpectrogram).unsqueeze(0)
                MelSpectrograms = torch.cat((MelSpectrograms, MelSpectrogram), 0)  # 应该是[clips, 9, 3, 224, 224]

    return MelSpectrograms


def video_decode(path, mode):
    """
    param path: 视频地址
    return: video_tensor
    """
    vid = cv2.VideoCapture(path)
    wid = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    hei = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    fnum = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    cnt = 0
    rval = vid.isOpened()
    while rval:
        if rval and cnt < fnum:
            rval, frame = vid.read()
            f = torch.tensor(frame)
            f = torch.unsqueeze(f, 0)

            if cnt == 0:
                video_tensor = f
            else:
                video_tensor = torch.cat((video_tensor, f), 0)
            cnt += 1
        else:
            break
    vid.release()

    return video_tensor, fps, fnum,


def temporal_sampling(frames, start_idx, end_idx, num_samples):  # 读一个规格化的数据
    """
    参数:
    frames : 数据帧, 格式 [T,C,H,W]
    start_idx : 截取点的开始帧序号
    end_idx : 截取点的结束帧序号
    num_samples : 要截取的帧数

    功能:
    给定一个视频数据帧，在一个指定帧段里均匀截取指定帧数
    """
    if type(start_idx) == int:
        index = torch.linspace(start_idx, end_idx, num_samples)  # 创建一个在start_idx到end_idx均匀分布的有num_samples个数的向量
        index = torch.clamp(index, 0, frames.shape[0] - 1).long()
        # 把刚擦创建的数字 规范化, 数字要在 0 与 frames.shape[0] - 1 之间
        new_frames = torch.index_select(frames, 0, index)  # 返回 index指出维度 选取的数据帧

        # TODO : 音频

        return new_frames
        # 格式[T, H, W]和[L, H]

    else:
        lens = len(start_idx)

        for idx in range(0, lens):
            index = torch.linspace(start_idx[idx], end_idx[idx], num_samples)
            # 创建一个在start_idx到end_idx均匀分布的有num_samples个数的向量
            index = torch.clamp(index, 0, frames.shape[0] - 1).long()
            # 把刚创建的数字 规范化, 数字要在 0 与 frames.shape[0] - 1 之间
            if idx == 0:
                new_frames = torch.unsqueeze(torch.index_select(frames, 0, index), 0)  # 返回 index指出维度 选取的数据帧
                # print(new_frames.shape)
            else:
                frame = torch.unsqueeze(torch.index_select(frames, 0, index), 0)
                # print(frames.shape)
                new_frames = torch.cat((new_frames, frame), 0)

        return new_frames  # 格式[clips, T, H, W]


def get_start_end_idx(video_size, clip_size, num_clips):  # 用来指出clip的头尾帧序号
    """
    参数
        video_size (int): 视频有多少帧
        clip_szie (int): clip多大
        num_clips (int):
            if num_clips = -1, 直接随机取
            If num_clips = 1,  取中间
            If num_clips = n,  均匀取n个, test才这么做

    Returns:
        start_idx (int): the start frame index.一个数或一组
        end_idx (int): the end frame index.一个数或一组
    """
    delta = max(video_size - clip_size, 0)  # 指出的区间段
    if num_clips == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
    else:
        if num_clips == 1:  # 取中间
            # Take the center clip if num_clips is 1.
            start_idx = math.floor(delta / 2)
            end_idx = start_idx + clip_size - 1
        else:  # n个clips 就均匀取法, 并返回一组开始与结束标记
            # Uniformly sample the clip with the given index.
            start_idx = []
            end_idx = []
            for idx in range(0, num_clips):
                start = idx * math.floor(delta / (num_clips - 1))
                start_idx.append(start)
                end_idx.append(start + clip_size - 1)

    return start_idx, end_idx


def decode(
        video_path,
        num_frames,
        mode,
        num_clips=-1,
        target_fps=30,
):
    """
    参数:
    video_path (str) : 视频路径
    num_frames (int) : 抽几帧
    stage : 模型的什么阶段，pretrain, fintune, test
    num_clips (int) : 切几个clips

    Returns:
        frames OR frames, audio
    """

    if mode == 'pretrain':  # pretrain和fintune才是用这个, 表示取全部的clips
        # print(video_path)

        video_tensor, fps, fnum = video_decode(video_path, mode)

        clip_size = fps * 3  # 取3秒

        start_idx, end_idx = get_start_end_idx(
            fnum,
            clip_size,
            num_clips,
        )

        frames = temporal_sampling(video_tensor, start_idx, end_idx, num_frames)

        audio_mel = audio_decode(video_path, start_idx, end_idx, num_frames)

        return frames, audio_mel

    else:
        video_tensor, fps, fnum = video_decode(video_path, mode)

        clip_size = num_frames / target_fps * fps  # 可以改？

        start_idx, end_idx = get_start_end_idx(
            fnum,
            clip_size,
            num_clips,  # 10个
        )

        frames = temporal_sampling(video_tensor, start_idx, end_idx, num_frames)

        return frames


class M2M(torch.utils.data.Dataset):  # modality to modality
    def __init__(
            self,
            # pattern setting
            mode,
            path_to_data_dir,
            # decode setting
            num_frames,
            num_clips,
            # crop setting
            crop_size,
            jitter_scales=(256, 320),
            jitter_scales_relative=[0.5, 1.0],
            jitter_aspect_relative=[0.75, 1.3333],
            # pretrain augmentation
            pretrain_rand_erase_prob=0.25,
            pretrain_rand_erase_mode="pixel",
            pretrain_rand_erase_count=1,
            pretrain_rand_flip=True,
            aa_type="rand-m7-n4-mstd0.5-inc1",

    ):
        self.mode = mode
        self.path_to_data_dir = path_to_data_dir,
        self.num_frames = num_frames
        self.num_clips = num_clips

        self.crop_size = crop_size

        self.pretrain_rand_flip = pretrain_rand_flip  # 是否随机翻转
        self.pretrain_rand_erase_prob = pretrain_rand_erase_prob
        self.pretrain_rand_erase_mode = pretrain_rand_erase_mode  # 随机擦除模式
        self.pretrain_rand_erase_count = pretrain_rand_erase_count  # 随机擦除数
        self.aa_type = aa_type
        self.jitter_aspect_relative = jitter_aspect_relative
        self.jitter_scales_relative = jitter_scales_relative
        self.jitter_scales = jitter_scales

        self.construct_loader()

    def construct_loader(self):
        file_name = {
            "pretrain": "train",
            "finetune": "train",
            "test": "val",
        }

        path_to_file = os.path.join(self.path_to_data_dir[0], "{}.txt".format(file_name[self.mode]))

        path_prefix = os.path.join(self.path_to_data_dir[0], "{}".format(file_name[self.mode]))

        self.path_to_videos = []


        with open(path_to_file, "r") as f:
            for path in f.read().splitlines():
                self.path_to_videos.append(os.path.join(path_prefix, path))  # 读取完整的视频路径文件, 并合成一个list


        f.close()

    def __getitem__(self, index):

        if self.mode in ["pretrain"]:
            frames, audio_mel = decode(
                self.path_to_videos[index],
                self.num_frames,
                self.mode,
                self.num_clips
            )
        else:
            frames = decode(
                self.path_to_videos[index],
                self.num_frames,
                self.mode,
                self.num_clips
            )

        # 0,1,2表示左、中、右裁剪, -1三个挑一个
        if self.mode in ["pretrain", "finetune"]:
            spatial_sample_index = -1
            min_scale, max_scale = self.jitter_scales

        else:
            spatial_sample_index = 1
            min_scale, max_scale, crop_size = (
                [self.crop_size] * 3
                if self.num_clips > 1
                else [self.jitter_scales[0]] * 2 + [self.crop_size]
            )

        if self.mode in ["pretrain"]:
            video, mel = self._aug_frame(frames,
                                         spatial_sample_index,
                                         min_scale,
                                         max_scale,
                                         self.crop_size,
                                         audio_mel
                                         )
            # print("video shape:{}".format(video.shape))
            # print("audio shape:{}".format(mel.shape))

            return video, mel

        else:
            video = self._corp_frame(frames,
                                     spatial_sample_index,
                                     min_scale,
                                     max_scale,
                                     self.crop_size,
                                     )

            return video,

    def _aug_frame(self, frames, spatial_sample_index, min_scale, max_scale, crop_size, audio=None):
        # 有且仅有pretrain和fintune可用
        frame_list = []
        audio_list = []
        lens = frames.shape[0]  # [N, T , C, H, W]

        for n in range(0, lens):
            frame = frames[n]
            mel = audio[n]
            # 一些随机增强
            aug_transform = create_random_augment(
                input_size=(frame.size(1), frame.size(2)),
                auto_augment=self.aa_type,
                interpolation="bicubic",
            )
            # T H W C -> T C H W.
            frame = frame.permute(0, 3, 1, 2)
            list_img = self._frame_to_list_img(frame)
            list_img = aug_transform(list_img)
            frame = self._list_img_to_frames(list_img)
            frame = frame.permute(0, 2, 3, 1)

            frame = utils.tensor_normalize(
                frame,
                (0.45, 0.45, 0.45),
                (0.225, 0.225, 0.225),
            )
            # T H W C -> C T H W.
            frame = frame.permute(3, 0, 1, 2)

            scl, asp = (
                self.jitter_scales_relative,
                self.jitter_aspect_relative,
            )
            relative_scales = (
                None
                if (self.mode not in ["pretrain", "finetune"] or len(scl) == 0)
                else scl
            )
            relative_aspect = (
                None
                if (self.mode not in ["pretrain", "finetune"] or len(asp) == 0)
                else asp
            )

            frame = utils.spatial_sampling(
                frame,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.pretrain_rand_flip,
                inverse_uniform_sampling=False,
                aspect_ratio=relative_aspect,
                scale=relative_scales,
                motion_shift=False,
            )

            # 随机像素擦除
            if self.pretrain_rand_erase_prob > 0.0:
                erase_transform = RandomErasing(
                    self.pretrain_rand_erase_prob,
                    mode=self.pretrain_rand_erase_mode,
                    max_count=self.pretrain_rand_erase_count,
                    num_splits=self.pretrain_rand_erase_count,
                    device="cpu",
                )
                if self.mode in ["pretrain"]:

                    frame = frame.permute(1, 0, 2, 3)  # [T, C, H, W]类似？
                    # print(frames.shape)
                    frame = erase_transform(frame)
                    # frame = frame.permute(1, 0, 2, 3)

                    # audio = audio.permute(1, 0, 2, 3)
                    # print(audio.shape)
                    mel = erase_transform(mel)
                    # audio = audio.permute(1, 0, 2, 3)

                    frame_list.append(frame)
                    audio_list.append(mel)

                elif self.mode in ["fintune"]:
                    frames = frames.permute(1, 0, 2, 3)  # [T, C, H, W]类似？
                    frames = erase_transform(frames)
                    frames = frames.permute(1, 0, 2, 3)

                    frame_list.append(frame)

        if self.mode in ["pretrain"]:
            frames = torch.stack(frame_list, dim=0)
            mel = torch.stack(audio_list, dim=0)

            return frames, mel

        elif self.mode in ["fintune"]:
            frames = torch.stack(frame_list, dim=0)

            return frames

    def _corp_frame(self, frames, spatial_sample_index, min_scale, max_scale, crop_size):
        # test用

        lens = frames.shape[0]  # [N, T , C, H, W]

        for n in range(0, lens):
            frame = frames[n]
            frame = utils.tensor_normalize(
                frame,
                (0.45, 0.45, 0.45),
                (0.225, 0.225, 0.225),
            )
            frame = frame.permute(3, 0, 1, 2)

            scl, asp = (
                self.jitter_scales_relative,
                self.jitter_aspect_relative,
            )
            relative_scales = (
                None
                if (self.mode not in ["pretrain", "finetune"] or len(scl) == 0)
                else scl
            )
            relative_aspect = (
                None
                if (self.mode not in ["pretrain", "finetune"] or len(asp) == 0)
                else asp
            )
            frame = utils.spatial_sampling(
                frame,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=False,
                inverse_uniform_sampling=False,
                aspect_ratio=relative_aspect,
                scale=relative_scales,
            )
        return frames  # [N, T, H, W]

    def _frame_to_list_img(self, frames):  # 数据增强训练才用
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        # print(len(self.path_to_videos))
        return len(self.path_to_videos)


import argparse
import numpy as np
import os
import torch
import MMAE.models_vit as modellist
from MMAE.Dataset import VMR_Dataset as VMR_Dataset
from MMAE.util.videotransforms import Transforms_valid as Transforms
from torch.utils.data import DataLoader


"""
Notation:  dataset里要写新的传递inference数据的接口
           本文件是作用是:可以在训练中当验证，也可以在完成后完成单次推理
           这里读取的数据因为写的dataset的原因 视频和音频 是 等数量的，想要不等数量需要重新写两个，使用两个分别的dataloder读取

"""
def get_args_parser():
    parser = argparse.ArgumentParser(description='Arguments of Inference')
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    parser.add_argument('--batch_size', type=int) # 最好大点，因为算相似度只在一个batch内算
    parser.add_argument('--datasetpath', default='/root/autodl-tmp/VMR_PRO/vaild', type=str)
    parser.add_argument('--log_dir', default='xxx', type=str)
    # TODO: 想要传入新的参数写在这里

def cosine_sim(im, s):  # 已修改，无问题
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())  # .mm() 矩阵乘法 ， .t()二维矩阵转置
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)  # margin = 1e-18
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)

    return sim


def forward_embed(video_data, audio_data, model_v, model_a):  # 调用模型，和得到计算结果

    video_embeds = model_v(video_data)
    audio_embeds = model_a(audio_data)

    return video_embeds, audio_embeds


def generate_scores(**kwargs):  # 生成分数的核心就是 cos相似度   # 已修改，无问题
    # compute image-sentence similarity
    vid_embeds = kwargs['vid_embeds']
    adu_embeds = kwargs['aud_embeds']
    scores = cosine_sim(vid_embeds, adu_embeds)  # s[i, j] i: im_idx, j: s_idx

    return scores


def evaluate_scores(dataloader, model_v, model_a):  # tst_reader是个dataloader ，       评估的是一个视频对n个批次音频的相似度   ， 第2级

    all_video_names, all_audio_names = [], []  # 名字
    all_scores = []

    for batch in dataloader:  # 这么读取是dataset写好的固定方式，我还没写
        print("进循环")
        video_names = batch['video_name']
        audio_names = batch['audio_name']
        video_data = batch['video']
        audio_data = batch['mel']

        video_feature, audio_feature = forward_embed(video_data, audio_data, model_v, model_a)
        embed = {'vid_embeds': video_feature, 'aud_embeds': audio_feature}

        score = generate_scores(**embed)  # video 对 audio 的score

        all_video_names.append(video_names)
        all_audio_names.append(audio_names)
        all_scores.append(score)  # 这是n个批次的分数

    return all_video_names, all_audio_names, all_scores


def evaluate(dataloader, model_v, model_a):  # 单一计算分数的函数   ，  # 已修改，无问题   # 第1级

    video_names, audio_names, all_scores = evaluate_scores(dataloader, model_v, model_a)  # 接收video, audio, 和对应分数矩阵（应该就是正方形的）

    ranking_list = []  # 创立排名矩阵

    lens = len(all_scores[0][0])  # 多少个audios
    size = len(all_scores[0])

    for i in range(0, size):
        ranking_branch = []
        for j in video_names:  # videonames遍历(可能存在问题)
            ranking_branch.append([])  # 加一行
            while len(all_scores[i][j]) != 0:
                max = 0
                k = 0
                index = 0

                for k in range(0, len(all_scores[i][j])):
                    if all_scores[i][j][k] >= max:
                        index = j
                        max = all_scores[i][j][k]

                ranking_branch[-1].append(audio_names[index])  # 写上audio名字

        ranking_list.append(ranking_branch)

    outs = {
        'video_names': video_names,
        'audio_names': audio_names,
        'ranking_': ranking_list,
    }

    # 你也可以用这个返回，但是没太大用

    # return video_names, audio_names, ranking_list


def assess(args, load=True):  # 综合的： 读取，计算，写入   # 未修改完全

    test_dataset = VMR_Dataset(args.datasetpath,
                               transforms= Transforms(224),
                               row=slice(0, None))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batchsize,
                             drop_last=True,
                             num_workers=8,
                             shuffle=False)

    model_v = modellist.__dict__[args.model]
    model_a = modellist.__dict__[args.model]
    if args.log_dir is not None and load == True:
        pass
        #TODO : 使用正确的模型存放地址与正确加载
        #model_path = os.path.join(log_dir, 'state', 'epoch{}.pt'.format(num_epoch))  # 拿出地址
        #model.load_state_dict(torch.load(model_path))  # 读取这个点的模型存档，写想读的存档点
        model_v.eval()
        model_a.eval()

        # eval_start()  # 我还没弄懂是干什么的

    video_names, audio_names, ranks = evaluate(test_loader, model_v, model_a)
    # 现在接收的是三维的数据[nums, batchsize], [nums, batchsize], [nums, batchsize, batchsize]

    # all_scores = np.concatenate(all_scores, axis=0)  # (n_video, n_audio) 二维数组， 每行对应的是不同的video，每列是对应的排好序的audio的分数
    # 作用不明，我未搞懂， 但是应该是写合在一起得分数矩阵

    # with open(os.path.join(log_dir, 'rank/epoch{}.txt'.format(num_epoch)), 'wb') as f:  # 打开想存的文件位置, 没有直接创建

    if not os.path.exists(os.path.join(args.log_dir, 'rank/epoch{}'.format(args.num_epoch))):  # 创建rank/epoch n 的文件，存单个np的数组
        os.mkdir(args.log_dir)

    vname = np.array(video_names)
    np.save(os.path.join(args.log_dir, 'rank/epoch{}_vname'.format(args.num_epoch)), vname)
    aname = np.array(audio_names)
    np.save(os.path.join(args.log_dir, 'rank/epoch{}_aname'.format(args.num_epoch)), aname)
    rank = np.array(ranks)
    np.save(os.path.join(args.log_dir, 'rank/epoch{}_rank'.format(args.num_epoch)), rank)

    #return {'batch_video_names': video_names, 'batch_audio_names': audio_names, 'batch_rank': ranks}

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()


    assess(args, load=True)

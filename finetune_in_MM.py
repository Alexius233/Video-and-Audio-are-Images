import argparse
import time
import datetime
import os
import torch
import torch.distributed as dist
import torch.nn as nn

from MMAE.util.M2M import VMR_Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from finetune_one_epoch import finetune_one_epoch
from MMAE.util.misc import NativeScalerWithGradNormCount as NativeScaler
import MMAE.models_vit as modellist
from MMAE.util.videotransforms import Transforms_finetune as Transforms
from torch.utils.tensorboard import SummaryWriter


def projector(inputsize):
    layers = []
    layers.append(nn.Linear(inputsize, 2048))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

class FinetuneModel(nn.Module):
    def __init__(self, args):
        super(FinetuneModel, self).__init__()
        self.videoencoder = modellist.__dict__[args.model]
        self.audioencoder = modellist.__dict__[args.model]

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        # TODO: projector的规格
        self.projector_a = projector(114514)
        self.projector_v = projector(114514)

        # load model
        model_path = os.path.join(args.loadpath, 'save_at_epoch{}.pt'.format(args.start_epoch))
        self.videoencoder.load_state_dict(torch.load(model_path))
        self.audioencoder.load_state_dict(torch.load(model_path))

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * args.batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(args.batch_size):
            mask[i, args.batch_size + i] = 0
            mask[args.batch_size + i, i] = 0
        return mask

    def forward_loss(self, afeature, vfeature):

        N = 2 * self.batch_size * self.world_size

        f = torch.cat((afeature, vfeature), dim=0)

        sim = self.similarity_f(f.unsqueeze(1), f.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, args.batch_size)
        sim_j_i = torch.diag(sim, -args.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


    def forward(self, imgs, mel):
        a = self.audioencoder(mel)
        v = self.videoencoder(imgs)

        A = self.projector_a(a)
        V = self.projector_v(v)

        loss = self.forward_loss(imgs, mel)

        return loss, A, V

def get_args_parser():
    parser = argparse.ArgumentParser(description='Arguments of Model')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--batch_size', type=int)   #batch_size per GPU
    parser.add_argument('--mixup_mode', type=str)
    parser.add_argument('--epochs', type=int)
    # Optimizer parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--blr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--start_epoch', default=0, type=int)
    # some pathes
    parser.add_argument('--logpath', default='/root/autodl-tmp', type=str)
    parser.add_argument('--datasetpath', default='/root/autodl-tmp/VMR_PRO/train', type=str)
    parser.add_argument('--savepath', default='/root/autodl-tmp/finetunemodelsave', type=str)
    parser.add_argument('--loadpath', default='/root/autodl-tmp/finetunemodelsave', type=str)
    # distributed
    parser.add_argument('--local_rank', default=-1, type=int) #命令行调用时由pytorch自动设置，我不用写
    # model control
    parser.add_argument('--model', default='vit_large_patch16', type=str)
    return parser


def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl') # nccl多GPU通信的功能，windows下不能用

    # 准备数据
    # DistributedSampler
    root = None #到时候写
    train_dataset = VMR_Dataset(args.datasetpath,
                                transforms=Transforms(224),
                                row=slice(0, None))
    train_sampler = torch.utils.data.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, sampler=train_sampler)

    # 声明tensorboard
    writer = SummaryWriter()
    #得到model并迁移到device上
    model = FinetuneModel(args)
    model.to(local_rank)


    # 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # 把model转成多GPU的实现数据并行

    # loss_scaler声明
    loss_scaler = NativeScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
        #torch.optim.SGD(model.parameters(), lr=args.lr)

    # 开始训练
    for epoch in range(args.epochs):
        if local_rank == 0:
            print("-------Training for epoch {}------".format(epoch))
        step = 1
        #开始训练的时间
        start_time = time.time()

        # sampler
        train_dataloader.sampler.set_epoch(epoch)  # 防止loss震荡

        #开始一个epoch的训练，
        finetune_one_epoch(model=model,
                        train_dataloader=train_dataloader,
                        optimizer=optimizer,
                        local_rank=local_rank,
                        loss_scaler=loss_scaler,
                        writer = writer,
                        step = step,
                        epoch = epoch,
                        args=args)

        #打印训练一个epoch花费的时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time(one epoch) : {}'.format(total_time_str))    #输出格式：Training time : 0:00:49
        # save model
        if epoch % 50 == 0 and epoch != 0:
            if dist.get_rank() == 0:
                torch.save(model.module.state_dict(), os.path.join(args.savepath, 'save_at_epoch{}.pt'.format(epoch)))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()


    main(args)
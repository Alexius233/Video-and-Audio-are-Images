import argparse
import time
import datetime
import os
import torch
import torch.distributed as dist

from MMAE.util.M2M import VMR_Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from train_one_epoch import train_one_epoch
from MMAE.util.misc import NativeScalerWithGradNormCount as NativeScaler
import MMAE.models_mae
from MMAE.util.videotransforms import Transforms_train as Transforms
from torch.utils.tensorboard import SummaryWriter



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
    parser.add_argument('--savepath', default='/root/autodl-tmp/pretrainmodelsave', type=str)
    parser.add_argument('--loadpath', default='/root/autodl-tmp/pretrainmodelsave', type=str)
    # distributed
    parser.add_argument('--local_rank', default=-1, type=int) #命令行调用时由pytorch自动设置，我不用写
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
    model = MaskedAutoencoderViT()
    model.to(local_rank)

    # load model
    if args.start_epoch != 0:
        model_path = os.path.join(args.loadpath, 'save_at_epoch{}.pt'.format(args.start_epoch))
        model.load_state_dict(torch.load(model_path))
        # 加载之前保存的参数, load_state_dict: 加载参数, torch_load: 读取文件内的参数，返回参数

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
        train_one_epoch(model=model,
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





import math
import sys
import torch

from MMAE.util.mix_function import Mixup
from MMAE.util import misc
from MMAE.util import lr_sched
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model: torch.nn.Module,
                    train_dataloader,
                    optimizer: torch.optim.Optimizer,
                    local_rank, loss_scaler,
                    writer,
                    step,
                    epoch,
                    args):
    # 设置model为train模式
    model.train()  # 库函数，针对BN和Dropout的设置
    optimizer.zero_grad()

    total_loss = 0
    for data in train_dataloader:
        #得到数据并迁移到cuda上
        imgs = data['video']
        mel = data['mel']
        mixup = Mixup(mode=args.mixup_mode)
        Mix = mixup(imgs, mel)  # [Batch_size, frame per video, channels, h, w]
        imgs = imgs.to(local_rank)
        mel = mel.to(local_rank)
        Mix = Mix.to(local_rank)

        with torch.cuda.amp.autocast():
            loss, _,_ = model(imgs, mel, Mix)

        loss_value =loss.item() #用.item()防止tensor无限叠加导致显存爆炸
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #反向传播、参数更新
        loss_scaler(loss, optimizer, parameters=model.parameters()) # loss_scaler还有一个bool类型的update_grad参数，没看懂
        optimizer.zero_grad()

        torch.cuda.synchronize()

        loss_value_reduce = misc.all_reduce_mean(loss_value) # 数据并行
        total_loss += loss_value

        # lr更新的trick
        lr_sched.adjust_learning_rate(optimizer, step / len(train_dataloader) + epoch, writer, step, args)

        if local_rank == 0:
            print("loss value is :{} in step {}".format(loss_value_reduce, step))
            writer.add_scalar("Loss/per_step", loss_value_reduce, step)  # Loss per step
        step += 1

        del Mix, mel, imgs

    if local_rank == 0:
        writer.add_scalar("Loss/per_epoch", total_loss / step, epoch) # Loss per Batch

    ############################### 用test_dataloader进行评估 ###############################
    # with torch.no_grad:
    #     # 设置model为eval模式
    #     model.eval()
    #     # 开始让总loss为0
    #     total_loss = 0
    #     for data in test_dataloader:
    #         imgs = data['video1']
    #         mel = data['mel']
    #         mixup = Mixup(mode=args.mixup_mode)
    #         Mix = mixup(imgs, mel)  # [Batch_size, frame per video, channels, h, w]
    #         imgs = imgs.to(local_rank)
    #         mel = mel.to(local_rank)
    #         Mix = Mix.to(local_rank)
    #
    #         loss, _, _ = model(imgs, mel, Mix)
    #
    #         loss_value = loss.item()
    #         if not math.isfinite(loss_value):
    #             print("Loss is {}, stopping evaluating".format(loss_value))
    #             sys.exit(1)
    #
    #         total_loss += loss_value
    #
    #     print("总损失为:{}".format(total_loss))
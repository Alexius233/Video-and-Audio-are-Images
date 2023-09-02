import os
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_PORT'] = '12345'
    local_rank = 0
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo')
    print(dist.get_world_size())    # 1

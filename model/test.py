import torch
from MMAE.util.mix_function import Mixup
import models_mae

imgs = torch.rand(2, 4, 3, 224, 224) # [Batch_size, frame per video, channels, h, w]
mel = torch.rand(2, 4, 3, 224, 224)

mixup = Mixup(mode="pixels")
Mix = mixup(imgs, mel)  # [Batch_size, frame per video, channels, h, w]

#############################

model = models_mae.mae_vit_base_patch16_dec512d8b()

x, y, z = model(imgs, mel, Mix)

print(Mix.shape)
from MMAE.util.Kinetics import Kinetics
from torch.utils.data import DataLoader

path_to_data_dir = None
sampling_rate = 4
num_frames = 9
repeat_aug = 2
jitter_aspect_relative = [0.75, 1.3333]
jitter_scales_relative = [0.5, 1.0]



dataset_train = Kinetics(
    mode="pretrain",
    path_to_data_dir=path_to_data_dir,
    sampling_rate=sampling_rate,
    num_frames=num_frames,
    train_jitter_scales=(256, 320),
    repeat_aug= repeat_aug,
    jitter_aspect_relative=jitter_aspect_relative,
    jitter_scales_relative=jitter_scales_relative,
    )

train_dataloader = DataLoader(dataset_train, batch_size=4, num_workers=4)

for data, _ in train_dataloader:
    print(data.shape)
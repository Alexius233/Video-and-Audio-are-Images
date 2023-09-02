from torchvision import transforms


def Transforms_train(size, s=1) : # TODO: 看看MAE的augmentation怎么写的
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
                                         transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          #transforms.RandomApply([color_jitter], p=0.8),
                                          #transforms.RandomGrayscale(p=0.2)
                                         # transforms.ToTensor()
                                         ])
    return data_transforms

def Transforms_finetune(size, s=1) :
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
                                         transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          #transforms.RandomApply([color_jitter], p=0.8),
                                          #transforms.RandomGrayscale(p=0.2)
                                         # transforms.ToTensor()
                                         ])
    return data_transforms

def Transforms_valid(size, s=1) : # TODO: 只需要resize或者裁剪
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
                                         transforms.RandomResizedCrop(size=size),

                                         # transforms.ToTensor()
                                         ])
    return data_transforms


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames = transform.random_crop(frames, crop_size)
        else:
            transform_func = (
                transform.random_resized_crop_with_shift
                if motion_shift
                else transform.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1
        frames = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames
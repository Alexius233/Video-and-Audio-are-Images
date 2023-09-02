import torch
import torchvision
from torch import nn
import numpy as np


def rand_bbox(img_shape, lam, margin=0., count=None):  # 根据λ值生成 大小固定 随机 方框，返回四个角坐标
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    """
    img_shape = 图像形状
    lam = cut部分占比
    margin = 生成的方框中心的偏移平衡系数：系数是保证box在一个范围内
    count = 生成的box数目
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)

    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)

    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)

    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):  # 根据λ值区间生成 大小非固定 随机 方框，返回四个坐标
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.
    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.
    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    """
    img_shape = 图片形状
    minmax = 掩码区域的占比率大小范围 minmax[0]是下限，minmax[1]是上限
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]

    if minmax[0] != minmax[1]:
        cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
        cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    else:
        cut_h = int(img_h * minmax[0])
        cut_w = int(img_w * minmax[0])
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):  # 部署两个函数返回四角坐标以及真λ值

    # λ因为 开根号和非整数原因存在实际值与输入值的偏差，或者minmax的随机性生成的值，这里重新计算返回实际值
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:  # 调用上面的混合策略函数，给图像混合
    """ Mixup/Cutmix that applies different params to each pixels, picture or whole batch
    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element: not right!!!
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """
    """
    mixup_alpha : >0 激活
    cutmix_alpha : >0 激活  这个需要  ！
    cutmix_minmax : >0 激活  这个需要 ！
    prob : 部署 cutmix 的概率概率 or pixels混的概率
    switch_prob : 部署 cutmix或mixup 的概率 ？
    mode : 混合的部署位置 ： batch, pair 或 element
    correct_lam : 真正的λ值
    label_smoothing : label平滑度
    num_classes : class数字
    """

    """
    功能:
    1. 实现pixels的混合，需要参数:prob,表示交换率
    2. 裁剪式混合，需要参数:cutmix_alpha = 1, swith_prob表示谁 被 混合的概率
    3. 叠加式混合，需要参数:mixup_alpha = 1
        ----可以选择混合的比例的概率区间:使用参数minmax -> 一个数组两个value [0.5,0.6]
            如果想固定混合的比例:设置minmax里minmax[0] = minmax[1]
            这里只能实现随机出一个窗口
    4. cutmix_alpha = mixup_alpha = 1,switch_prob决定应用谁的概率
    5. 三种模式:pixels, picture, video
    """

    def __init__(self, mixup_alpha=0., cutmix_alpha=1., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='image', correct_lam=True):
        # 参数初始化
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_image(self, num):  # image 级交换 的 参数
        lam = np.ones(num, dtype=np.float32)
        use_cutmix = np.zeros(num, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:  # 都大于0，两个都部署
                use_cutmix = np.random.rand(num) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    # beta分布:定义在 (0,1) 区间的连续概率分布 。
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=num),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=num))
            elif self.mixup_alpha > 0.:  # 只部署一个, 只有1出现lam_mix才能有数值
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=num)
            elif self.cutmix_alpha > 0.:  # 只部署一个, 只有1出现lam_mix才能有数值
                use_cutmix = np.ones(num, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=num)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(num) < self.mix_prob, lam_mix.astype(np.float32), lam)
            # where ： 在满足一定条件下，满足条件输出lam_mix，不满足输出lam
        return lam, use_cutmix

    def _params_per_video(self):  # 视频 级交换 的 参数
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)  # 只生成一个混合方式
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _params_per_pixels(self): # pixels级混合参数
        lens = 224 * 224
        num = np.ones(lens)
        exrate = int(self.mix_prob * lens)
        num[:exrate] = 0
        np.random.shuffle(num)

        return  num, self.mix_prob


    def _mix_image(self, x, y):  # 一个图片一个混合样式
        # 格式: [BS, N, 3, H, W]
        batch_size = x.shape[0]
        num = x.shape[1]  # 一个视频几张
        lam_batch, use_cutmix = self._params_per_image(num)
        x_orig = x.clone()  # 保持一个源数据
        y_orig = y.clone()
        x_operate = x.clone()
        y_operate = y.clone()
        chooseprob = np.random.rand(1) # 表示谁混谁的概率
        for n in range(batch_size):
            for i in range(0, num):
                lam = lam_batch[i]  # 取出随机生成的掩码率
                if lam != 1.:
                    if use_cutmix[i]:
                        (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                            x[n,i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)

                        if chooseprob < self.switch_prob:
                            x_operate[n, i, :, yl:yh, xl:xh] = y_orig[n, i, :, yl:yh, xl:xh]  # 模态间交换x插入y
                        else:
                            y_operate[n, i, :, yl:yh, xl:xh] = x_orig[n, i, :, yl:yh, xl:xh]
                            x_operate[n, i] = y_operate[n, i]
                        lam_batch[i] = lam  # 算出真实的掩码率
                    else:
                        x_operate[n][i] = x_orig[n][i] * lam + y_orig[n][i] * (1 - lam)

        mixdata = x_operate

        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1), mixdata

    def _mix_video(self, x, y):  # video内遵从一个混合样式
        lam, use_cutmix = self._params_per_video()
        batch_size = x.shape[0]
        num = x.shape[1]  # 一个视频几张
        x_orig = x.clone()
        y_orig = y.clone()
        x_operate = x.clone()
        y_operate = y.clone()
        chooseprob = np.random.rand(1)  # 表示谁混谁的概率
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            for n in range(0, batch_size):
                if chooseprob < self.switch_prob:
                    x_operate[n, :, :, yl:yh, xl:xh] = y_orig[n, :, :, yl:yh, xl:xh]

                else:
                    y_operate[n, :, :, yl:yh, xl:xh] = x_orig[n, :, :, yl:yh, xl:xh]
                    x_operate[n]= y_operate[n]
        else:
            for i in range(0, batch_size):
                for n in range(num):
                    x_operate[i][n] = x_orig[i][n] * lam + y_orig[i][n] * (1 - lam)

        mixdata = x_operate

        return lam, mixdata

    def _mix_pixel(self, x, y):
        num, lam = self._params_per_pixels()
        rate = torch.tensor(num).reshape(shape=(224,224))
        x_operate = x * rate + x * (1- rate)

        return lam, x_operate



    def __call__(self, x, y): # x图,y音,
        assert len(x) % 2 == 0, 'Batch size should be even when using this'

        if self.mode == 'image':
            lam, mixdata = self._mix_image(x, y)
        if self.mode == 'video':
            lam, mixdata = self._mix_video(x, y)
        else:
            lam, mixdata = self._mix_pixel(x, y)
        return mixdata  # 想看λ可以返回

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

#from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block

from MMAE.util.pos_embed import get_2d_sincos_pos_embed




class PatchEmbed(nn.Module):  # 还没改，audiomae抄来的
    """ Image to Patch Embedding
    """
    # 8张图一个
    # 输入格式 [batch_size, 3, H, W*9]
    def __init__(self, img_size=(224, 224*4), patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()


        num_patches = (img_size[1] // patch_size) * (img_size[0] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# 初步设想： 一张图 [C, H, W], n张图[C, H, W*n]

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=(224, 224*4), patch_size=16, in_chans=3,
                 embed_dim=768, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            #Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_img = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch, image的MLP
        self.decoder_pred_mel = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # mel的MLP
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #[-1] : 取最后一个
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def datareshape(self, imgs, mel, mix):
        # inputsize: [N, 9, 3, H, W]
        # mid: [N, 3, H, W, 9]
        # ouputsize: [N, 3, H, W*9]

        x = torch.einsum('ntchw->nchwt', imgs)
        x = x.reshape(shape=(x.shape[0], x.shape[1], x.shape[2], x.shape[3]*x.shape[4]))
        y = torch.einsum('ntchw->nchwt', mel)
        y = y.reshape(shape=(y.shape[0], y.shape[1], y.shape[2], y.shape[3] * y.shape[4]))
        z = torch.einsum('ntchw->nchwt', mix)
        z = z.reshape(shape=(z.shape[0], z.shape[1], z.shape[2], z.shape[3] * z.shape[4]))

        return x.float(), y.float(), z.float()

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W*9)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size
        #print(imgs.shape)
        #assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W*8)
        """
        p = self.patch_embed.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x_orig = self.decoder_norm(x)


        # predictor projection
        #print(x.shape)
        x = self.decoder_pred_img(x_orig)
        y = self.decoder_pred_mel(x_orig)

        # remove cls token
        x = x[:, 1:, :]
        y = y[:, 1:, :]

        return x, y

    def forward_loss(self, imgs, mels, pred_imgs, pred_mels, mask):   # 输入两个imgs和mel
        """
        imgs: [N, 3, H, W]
        mel:  [N, 3, H, W]
        target
        pred_imgs: [N, L, p*p*3]
        pred_mels:  [N, L, p*p*3]
        pre: [N, L, p*p*3*2]       # 整好了塞进来
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        """
        target = self.patchify(imgs)
        target = torch.cat((target, self.patchify(mel)), 2) # [N, L, p*p*3*2]

        pred = torch.cat((pred_imgs, pred_mel),2)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
        """
        target_imgs = self.patchify(imgs)
        target_mels = self.patchify(mels)

        if self.norm_pix_loss:
            mean_imgs = target_imgs.mean(dim=-1, keepdim=True)
            var_imgs = target_imgs.var(dim=-1, keepdim=True)
            target_imgs = (target_imgs - mean_imgs) / (var_imgs + 1.e-6) ** .5

            mean_mels = target_mels.mean(dim=-1, keepdim=True)
            var_mels = target_mels.var(dim=-1, keepdim=True)
            target_mels = (target_mels - mean_mels) / (var_mels + 1.e-6) ** .5

        loss_imgs = (pred_imgs - target_imgs) ** 2
        loss_mels = (pred_mels - target_mels) ** 2
        loss_imgs = loss_imgs.mean(dim=-1)
        loss_mels = loss_mels.mean(dim=-1)

        mask = mask.view(loss_imgs.shape)

        loss_imgs = (loss_imgs * mask).sum() / mask.sum()
        loss_mels = (loss_mels * mask).sum() / mask.sum()
        loss = loss_mels + loss_imgs

        return loss

    def forward(self, imgs, mel, Mix, mask_ratio=0.75):
        imgs, mel, Mix = self.datareshape(imgs, mel, Mix)
        latent, mask, ids_restore = self.forward_encoder(Mix, mask_ratio)
        pred_imgs, pred_mel = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3*2]
        pred = torch.cat((pred_imgs, pred_mel), 2)
        loss = self.forward_loss(imgs, mel, pred_imgs, pred_mel, mask)
        return loss, pred, mask
        # pred是tokens化的结果，想要原图使用unpatchify还原


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

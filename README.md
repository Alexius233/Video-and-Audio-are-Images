# CMMixer

An official implement of paper **''Video and Audio are Images: A Cross-Modal Mixer for Original Data on Video-Audio Retrieval''**

Full version in [[PDF]](https://arxiv.org/abs/2308.13820).

![](pic/main.png) 

## Abstract
Cross-modal retrieval has become popular in recent years, particularly with the rise of multimedia. Generally, the information from each modality exhibits distinct representations and semantic information, which makes feature tends to be in separate latent spaces encoded with dual-tower architecture and makes it difficult to establish semantic relationships between modalities, resulting in poor retrieval performance. To address this issue, we propose a novel frame- work for cross-modal retrieval which consists of a cross-modal mixer, a masked autoencoder for pre-training, and a cross-modal retriever for downstream tasks. In specific, we first adopt cross-modal mixer and mask modeling to fuse the original modality and eliminate redundancy. Then, an encoder-decoder architecture is applied to achieve a fuse-then-separate task in the pre-training phase. We feed masked fused representations into the encoder and reconstruct them with the decoder, ultimately separating the original data of two modalities. In downstream tasks, we use the pre-trained encoder to build the cross-modal retrieval method. Extensive experiments on 2 real-world datasets show that our approach outperforms previous state-of-the-art methods in video-audio matching tasks, improving retrieval accuracy by up to 2×. Furthermore, we prove our model performance by transferring it to other downstream tasks as a universal model.

## Requirements
```
- torch == 1.10.0+cu113
- torchvision == 0.11.1+cu113
- timm  == 0.3.2 
- numba == 0.48.0
- numpy == 1.21.4
- opencv-python == 4.6.0.66
- librosa == 0.9.2
```

## 

An example of training script:
```
python -m torch.distributed.launch --nproc_per_node 8 --use_env train_model.py --num_workers 32 --batch_size 8 --epochs 400 --lr 2e-4 --warmup_epochs 40 --mixup_mode image
```



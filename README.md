### SPN: Fully Context-Aware Image Inpainting with a Learned Semantic Pyramid
Code for Fully Context-Aware Image Inpainting with a Learned Semantic Pyramid, submitted to IEEE.

This project is for our new inpainting method SPN which has been submitted to IEEE under peer review. This work is an extension version of our previous work [SPL (IJCAI'21)](https://github.com/WendongZh/SPL) and the code and pre-trained models will be uploaded soon. Thanks for your intrests!

### Introduction:
Briefly speaking, in this work, we still focus on the key insight that learning semantic priors from specific pretext tasks can benefit image inpainting, and we further strengthen the modeling of the learned priors in this work from the following aspects:
1) We exploit multi-scale semantic priors in a feature pyramid manner to achieve consistent understanding of both gloabl and local context. The image generator is also improved to incorporate the prior pyramid.
2) We extend our prior learned in a probabilistic manner which enables our method to handle probabilistic image inpainting problem.
3) Besides, more analyses of the learned prior pyramid and the choices of the semantic supervision are provided in our experiment part.

## Prerequisites (same with SPL)
- Python 3.7
- PyTorch 1.8 (1.6+ may also work)
- NVIDIA GPU + CUDA cuDNN
- [Inplace_Abn](https://github.com/mapillary/inplace_abn) (only needed for training our model, used in [ASL_TRresNet](https://github.com/Alibaba-MIIL/ASL) model)
- torchlight (We only use it to record the printed information. You can change it as you want.)

## Datasets
We use [Places2](http://places2.csail.mit.edu), [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Paris Street-View](https://github.com/pathak22/context-encoder) datasets for determinstic image inpainting which is same with [SPL](https://github.com/WendongZh/SPL), and [CelebA-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset is used for probabilistic image inpainting. We also used the irregular mask provided by [Liu et al.](https://arxiv.org/abs/1804.07723) which can be downloaded from [their website](https://nv-adlr.github.io/publication/partialconv-inpainting). For the detailed processes of these datasets please refer to [SPL](https://github.com/WendongZh/SPL).

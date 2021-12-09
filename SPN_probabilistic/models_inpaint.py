import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import torchvision.models as models
from torchvision.models import resnet50
from networks import G_Net, VGG19, D_Net_SPADE
import torch.nn.functional as F
from src.models import create_model

class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class VGGLoss_withMS(nn.Module):
    def __init__(self):
        super(VGGLoss_withMS, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y, mask):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss_all = 0
        loss_MS = 0
        for i in range(len(x_vgg)):
            loss_all += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        for i in range(len(x_vgg)):
            x_vgg_tmp = x_vgg[i]
            B, _, H, W = x_vgg_tmp.size()
            x_vgg_tmp_1, x_vgg_tmp_2 = torch.split(x_vgg_tmp, int(B/2), 0)
            mask_tmp = F.interpolate(mask, H)
            loss_MS += self.criterion(x_vgg_tmp_1*mask_tmp, x_vgg_tmp_2*mask_tmp) / torch.mean(mask_tmp)
        return loss_all, loss_MS 

class InpaintingModel(nn.Module):
    def __init__(self, g_lr, d_lr, l1_weight, gan_weight, TRresNet_path=None, iter=0, threshold=None, device=None):
        super(InpaintingModel, self).__init__()

        self.generator = G_Net(input_channels=3, residual_blocks=8, threshold=threshold)
        self.discriminator = D_Net_SPADE(in_channels=3, use_sigmoid=False)

        if TRresNet_path is not None:
            # import pretrained tresnet_xL
            state_xL = torch.load(TRresNet_path, map_location='cpu')
            pretrained_model = state_xL['model']
            self.tresnet_xL_hold = create_model('tresnet_l')
            model_dict = self.tresnet_xL_hold.state_dict()
            new_dict = {k: v for k, v in pretrained_model.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.tresnet_xL_hold.load_state_dict(model_dict)

        self.l1_loss = nn.L1Loss()
        self.l1_loss_feature = nn.L1Loss(reduction='none')
        self.adversarial_loss = AdversarialLoss('hinge')
        self.vgg_loss = VGGLoss_withMS()

        self.g_lr, self.d_lr = g_lr, d_lr

        self.l1_weight, self.gan_weight = l1_weight, gan_weight

        self.global_iter = iter

    def make_DPP(self, local_rank):

        self.generator = torch.nn.parallel.DistributedDataParallel(self.generator,
                                                                    device_ids=[local_rank])
 
        self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator,
                                                                        device_ids=[local_rank])


    def make_optimizer(self):
        # ddp add here
        self.gen_optimizer = optim.Adam(
            [{'params': self.generator.parameters(), 'lr': float(self.g_lr)},],
            lr=float(self.g_lr),
            betas=(0., 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(self.d_lr),
            betas=(0., 0.9)
        )



# if __name__ == '__main__':

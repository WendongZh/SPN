from __future__ import print_function
import argparse
from math import log10
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# from module_util import *
from dataset import build_dataloader
import pdb
import socket
import time
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from models_inpaint import InpaintingModel
import torchlight
import torch.distributed as dist
from bilinear import crop_bbox_batch, uncrop_bbox
import cv2



# Training settings
parser = argparse.ArgumentParser(description='SPN')
parser.add_argument('--bs', type=int, default=8, help='training batch size')
parser.add_argument('--dataset', type=str, default='celeba-hq', help='used dataset')
parser.add_argument('--input_size', type=int, default=256, help='input image size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=70, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=67454, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--img_flist', type=str, default='shuffled_train.flist')
parser.add_argument('--mask_flist', type=str, default='all.flist')
parser.add_argument('--model_type', type=str, default='SPN')
parser.add_argument('--threshold', type=float, default=0.8, help='defaule mask threshold from RN model')
parser.add_argument('--val_prob_num', type=int, default=10, help='we use the first 50 images to evaluate our model during the training phase')
parser.add_argument('--lr_deacy_epoch', type=int, default=49, help='start epoch to deacy the learning rate')
parser.add_argument('--prior_cut_epoch', type=int, default=59, help='only for Paris datset')
parser.add_argument('--pretrained_sr', default='../weights/xx.pth', help='pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='./checkpoints/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='./SPN_probabilistic', help='Location to save checkpoint models')
parser.add_argument('--print_interval', type=int, default=400, help='how many steps to print the results out') 
parser.add_argument('--render_interval', type=int, default=600, help='how many steps to save a checkpoint') 
parser.add_argument('--l1_weight', type=float, default=10.0)
parser.add_argument('--gan_weight', type=float, default=1.0)
parser.add_argument('--with_test', default=False, action='store_true', help='Train with testing?')
parser.add_argument('--test', default=False, action='store_true', help='Test model')
parser.add_argument('--test_mask_flist', type=str, default='mask1k.flist')
parser.add_argument('--test_img_flist', type=str, default='val1k.flist')
parser.add_argument('--test_mask_index', type=str, default='selected_mask_fortest')
parser.add_argument('--TRresNet_path', type=str, default='./pretrained_TRresNet/')
parser.add_argument("--local_rank", default=0, type=int)

opt = parser.parse_args()

hostname = str(socket.gethostname())
opt.save_folder += opt.prefix

# will read env master_addr master_port world_size
torch.distributed.init_process_group(backend='nccl', init_method="env://")
opt.world_size = dist.get_world_size()
opt.rank = dist.get_rank()
# args.local_rank = int(os.environ.get('LOCALRANK', args.local_rank))
opt.total_batch_size = (opt.bs) * dist.get_world_size()
print("use total_batch_size:%s, world_size:%s rank:%s local_rank:%s" %(opt.total_batch_size, opt.world_size, opt.rank, opt.local_rank))

# init cuda env
cudnn.benchmark = True
torch.cuda.set_device(opt.local_rank)

if not os.path.exists(opt.save_folder):
    if opt.rank == 0:
        os.makedirs(opt.save_folder)

print(opt)


def train(epoch, FM_weight):
    iteration, avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss = 0, 0, 0, 0, 0
    last_l1_loss, last_gan_loss, cur_l1_loss, cur_gan_loss = 0, 0, 0, 0
    avg_vgg_loss, avg_kl_loss, avg_MS_loss, avg_FM_loss = 0,0,0,0
    avg_FD_loss = 0
    model.train()
    model.tresnet_xL_hold.eval()
    t0 = time.time()
    t_io1 = time.time()
    normal_distribution = torch.distributions.normal.Normal(torch.zeros(opt.bs, 128, 1, 1).cuda(), torch.ones(opt.bs, 128, 1, 1).cuda())
    for batch in training_data_loader:
        gt_512, gt, mask, index = batch
        t_io2 = time.time()
        if cuda:
            gt = gt.cuda(non_blocking=True)
            gt_512 = gt_512.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            mask = torch.mean(mask, 1, keepdim=True)
            gt_double = gt.repeat(2, 1, 1, 1)

        with torch.no_grad():
            mask_512 = F.interpolate(mask, 512)
            gt_256_masked = gt * (1.0 - mask) + mask
            gt_512_masked = F.interpolate(gt_256_masked, 512)
            img_f_full = model.tresnet_xL_hold(gt_512)
            img_f_full = [f_tmp.repeat(2, 1, 1, 1) for f_tmp in img_f_full]

        # img_f size is , 64, 64
        mask_64 = F.interpolate(mask, 64)
        mask_128 = F.interpolate(mask, 128)

        prediction, img_f_pred, z_distribution = model.generator(gt, mask, gt_512_masked, mask_512, True)
        img_pred_1, img_pred_2 = torch.split(prediction, gt.size(0), 0)
        img_pred_1 = img_pred_1 * mask + gt * (1 - mask)
        img_pred_2 = img_pred_2 * mask + gt * (1 - mask)

        # Compute Loss
        g_loss, d_loss, d_loss_objs = 0, 0, 0

        d_real, real_feature = model.discriminator(gt)
        real_feature = [f_tmp.repeat(2, 1, 1, 1) for f_tmp in real_feature]
        d_fake, _ = model.discriminator(prediction.detach())
        d_real_loss = model.adversarial_loss(d_real, True, True)
        d_fake_loss = model.adversarial_loss(d_fake, False, True)
        d_loss += (d_real_loss + d_fake_loss) / 2

        # Backward
        model.dis_optimizer.zero_grad()
        d_loss.backward()
        model.dis_optimizer.step()

        g_fake, fake_feature = model.discriminator(prediction)
        g_gan_loss = model.adversarial_loss(g_fake, True, False)
        g_loss += model.gan_weight * g_gan_loss
        g_l1_loss = torch.mean(model.l1_loss_feature(prediction, gt_double) * (1 + 2 * mask.repeat(2, 1, 1, 1)))
        img_f_full_256 = F.interpolate(img_f_full[0], 256)
        g_l1_FM_loss = torch.mean(model.l1_loss_feature(img_f_pred[0], img_f_full_256) * (1 + 3 * mask.repeat(2, 1, 1, 1)))
        g_l1_FM_loss = g_l1_FM_loss + torch.mean(model.l1_loss_feature(img_f_pred[1], img_f_full[1]) * (1 + 3 * mask_128.repeat(2, 1, 1, 1)))
        g_l1_FM_loss = g_l1_FM_loss + torch.mean(model.l1_loss_feature(img_f_pred[2], img_f_full[2]) * (1 + 3 * mask_64.repeat(2, 1, 1, 1)))
        # kl loss
        g_kl_loss = torch.mean(torch.distributions.kl_divergence(normal_distribution, z_distribution))
        # vgg loss
        g_vgg_loss, g_MS_loss = model.vgg_loss(prediction, gt_double, mask)
        # MS loss
        g_MS_loss = g_MS_loss
        eps = 1 * 1e-5
        g_MS_loss = 1 / (g_MS_loss + eps)
        # Feature matching loss
        g_FD_loss = 0
        for i in range(len(fake_feature)):
            g_FD_loss += model.l1_loss(fake_feature[i], real_feature[i].detach()) / len(fake_feature)
        g_loss += 10* g_FD_loss
   
        g_loss += model.l1_weight * g_l1_loss + g_l1_FM_loss*FM_weight + g_kl_loss*0.05 + g_MS_loss + model.l1_weight*g_vgg_loss # 

        model.gen_optimizer.zero_grad()
        g_loss.backward()
        model.gen_optimizer.step()

        reduced_g_l1_loss = reduce_tensor(g_l1_loss)
        reduced_g_gan_loss = reduce_tensor(g_gan_loss)
        reduced_g_loss = reduce_tensor(g_loss)
        reduced_d_loss = reduce_tensor(d_loss)
        reduced_g_l1_FM_loss = reduce_tensor(g_l1_FM_loss)
        reduced_g_vgg_loss = reduce_tensor(g_vgg_loss)
        reduced_g_kl_loss = reduce_tensor(g_kl_loss)
        reduced_g_MS_loss = reduce_tensor(g_MS_loss)
        reduced_g_FD_loss = reduce_tensor(g_FD_loss)

        avg_l1_loss += reduced_g_l1_loss.data.item()
        avg_gan_loss += reduced_g_gan_loss.data.item()
        avg_g_loss += reduced_g_loss.data.item()
        avg_d_loss += reduced_d_loss.data.item()
        avg_FM_loss += reduced_g_l1_FM_loss.data.item()
        avg_MS_loss += reduced_g_MS_loss.data.item()
        avg_vgg_loss += reduced_g_vgg_loss.data.item()
        avg_kl_loss += reduced_g_kl_loss.data.item()
        avg_FD_loss += reduced_g_FD_loss.data.item()


        model.global_iter += 1
        iteration += 1
        t1 = time.time()
        td, t0 = t1 - t0, t1

        if iteration % opt.print_interval == 0:
           
            if opt.rank == 0:
                torchlight_write.print_log(
                    "=> Epoch[{}]({}/{}): Avg l1 loss: {:.6f} | Avg FD loss: {:.6f} | Avg FM loss: {:.6f} | Avg vgg loss: {:.6f} | Avg MS loss: {:.6f} | Avg kl loss: {:.6f} | G loss: {:.6f} | Avg D loss: {:.6f} || Timer: {:.4f} sec. | IO: {:.4f}".format(
                        epoch, iteration, len(training_data_loader), avg_l1_loss / opt.print_interval, avg_FD_loss / opt.print_interval, avg_FM_loss / opt.print_interval, avg_vgg_loss / opt.print_interval,
                                                                     avg_MS_loss / opt.print_interval, avg_kl_loss / opt.print_interval, avg_g_loss / opt.print_interval,
                                                                     avg_d_loss / opt.print_interval, td, t_io2 - t_io1))

            avg_g_loss, avg_d_loss, avg_l1_loss, avg_gan_loss = 0, 0, 0, 0
            avg_vgg_loss, avg_kl_loss, avg_MS_loss, avg_FM_loss = 0,0,0,0
            avg_FD_loss = 0

        t_io1 = time.time()

        if opt.rank == 0 and iteration % opt.render_interval == 0:

            render(epoch, iteration, mask, [img_pred_1.detach(), img_pred_2.detach()], gt)


def dynamic_weigh(last_losses, cur_losses, T=20):
    # input lists
    last_losses, cur_losses = torch.Tensor(last_losses), torch.Tensor(cur_losses)
    w = torch.exp((cur_losses / last_losses) / T)
    return (last_losses.size(0) * w / torch.sum(w)).cuda()

def render(epoch, iter, mask, output, gt, state='train'):

    for i in range(3):

        name_pre = 'render/'+state+str(epoch)+'_'+str(iter)+'_'+str(i)+'_'
        # input: (bs,3,256,256)
        input = gt * (1 - mask) + mask
        input = input[i].permute(1,2,0).cpu().numpy()
        io.imsave(name_pre+'input.png', (input*255).astype(np.uint8))

        # mask: (bs,1,256,256)
        mask_tmp = mask[i,0].cpu().numpy()
        io.imsave(name_pre+'mask.png', (mask_tmp*255).astype(np.uint8))

        # output: (bs,3,256,256)
        output_tmp = output[0][i].permute(1,2,0).cpu().numpy()
        io.imsave(name_pre+'output_0.png', (output_tmp*255).astype(np.uint8))

        output_tmp = output[1][i].permute(1,2,0).cpu().numpy()
        io.imsave(name_pre+'output_1.png', (output_tmp*255).astype(np.uint8))

        # gt: (bs,3,256,256)
        gt_tmp = gt[i].permute(1,2,0).cpu().numpy()
        io.imsave(name_pre+'gt.png', (gt_tmp*255).astype(np.uint8))

def test(gen, dataloader, epoch):
    model = gen.eval()
    # model_yolo = detector.eval()
    psnr = 0
    count_psnr = 0
    for batch in dataloader:
        gt_512_batch, gt_batch, mask_batch, index = batch
        # gt, objs, boxes, masks_obj, obj_num_in_img, mask, obj_to_img, edge_64 = batch
        t_io2 = time.time()
        if cuda:
            gt_batch = gt_batch.cuda(non_blocking=True)
            gt_512_batch = gt_512_batch.cuda(non_blocking=True)
            mask_batch = mask_batch.cuda(non_blocking=True)
            mask_batch = torch.mean(mask_batch, 1, keepdim=True)

        with torch.no_grad():
            mask_512 = F.interpolate(mask_batch, 512)
            gt_256_masked = gt_batch * (1.0 - mask_batch) + mask_batch
            gt_512_masked = F.interpolate(gt_256_masked, 512)

            prediction, _, _ = model.generator(gt_batch, mask_batch, gt_512_masked, mask_512, True)
            img_pred_1, img_pred_2 = torch.split(prediction, gt_batch.size(0), 0)
            img_pred_1 = img_pred_1 * mask_batch + gt_batch * (1 - mask_batch)
            img_pred_2 = img_pred_2 * mask_batch + gt_batch * (1 - mask_batch)

        if opt.rank == 0:
            render(epoch, 0, mask_batch, [img_pred_1.detach(), img_pred_2.detach()], gt_batch, state='test')
        for i in range(gt_batch.size(0)):
            gt, pred_1, pred_2 = gt_batch[i], img_pred_1[i], img_pred_2[i]
            psnr += compare_psnr(pred_1.permute(1,2,0).cpu().numpy(), gt.permute(1,2,0).cpu().numpy(),\
            data_range=1)
            psnr += compare_psnr(pred_2.permute(1,2,0).cpu().numpy(), gt.permute(1,2,0).cpu().numpy(),\
            data_range=1)
            count_psnr += 2
        break

    return psnr / count_psnr

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= opt.world_size
    return rt

def checkpoint(epoch):
    model_out_path = opt.save_folder+'/'+'x_'+hostname + \
        opt.model_type+"_"+opt.prefix + "_bs_{}_epoch_{}.pt".format(opt.bs, epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def Kmeans_map(x):
    # x represents the input tensor
    c, h, w = x.size()
    x = x.view(c, h*w).contiguous().transpose(0, 1)
    x = x.cpu().numpy()
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    # clusters number K is set to 8
    K = 8
    compactness, labels, centers = cv2.kmeans(x, K, None, criteria, 10, flags)
    
    img_output = labels.reshape((h, w))
    img_output = np.uint8((img_output /8) * 255)
    im_color = cv2.applyColorMap(img_output, cv2.COLORMAP_JET) 
    return im_color

if __name__ == '__main__':
    # if opt.tb:
    #     writer = SummaryWriter()

    # Set the GPU mode
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    # Set the random seed
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed_all(opt.seed)

    # Model
    model = InpaintingModel(g_lr=opt.lr, d_lr=(opt.lr), l1_weight=opt.l1_weight, gan_weight=opt.gan_weight, TRresNet_path=opt.TRresNet_path, iter=0, threshold=opt.threshold)

    model = model.cuda()
    model.make_DPP(opt.local_rank)
    model.make_optimizer()

    # Load the pretrain model.
    if opt.pretrained:
        model_name = os.path.join(opt.pretrained_sr)
        print('pretrained model: %s' % model_name)
        if os.path.exists(model_name):
            loc = 'cuda:{}'.format(opt.local_rank)
            # pretained_model = torch.load(model_name, map_location=lambda storage, loc: storage)
            pretained_model = torch.load(model_name, map_location=loc)
            model.load_state_dict(pretained_model)
            print('Pre-trained model is loaded.')
            print(' Current: G learning rate:', model.g_lr, ' | L1 loss weight:', model.l1_weight, \
                  ' | GAN loss weight:', model.gan_weight)

    if opt.rank == 0:
        save_path = opt.save_folder + '/'
        torchlight_write = torchlight.IO(
            save_path,
            save_log=True,
            print_log=True
        )


    # Datasets
    print('===> Loading datasets')
    training_data_loader, sampler = build_dataloader(
        dataset_name=opt.dataset,
        flist=opt.img_flist,
        mask_flist=opt.mask_flist,
        test_mask_index = opt.test_mask_index,
        augment=True,
        training=True,
        input_size=opt.input_size,
        batch_size=opt.bs,
        num_workers=opt.threads,
        shuffle=True,
        world_size=opt.world_size,
        rank=opt.rank
    )
    print('===> Loaded datasets')

    if opt.test or opt.with_test:
        test_data_loader, _ = build_dataloader(
            dataset_name=opt.dataset,
            flist=opt.test_img_flist,
            mask_flist=opt.test_mask_flist,
            test_mask_index = opt.test_mask_index,
            augment=False,
            training=False,
            input_size=opt.input_size,
            batch_size=opt.val_prob_num,
            num_workers=opt.threads,
            shuffle=False,
            world_size=opt.world_size,
            rank=opt.rank
        )
        print('===> Loaded test datasets')

    if opt.test:
        test_psnr = test(model, test_data_loader)
        os._exit(0)

    # Start training
    best_psnr = 0.0
    FM_initial = 1.0
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        sampler.set_epoch(epoch)
        train(epoch, FM_initial)

        if epoch > opt.lr_deacy_epoch:
            for param_group in model.gen_optimizer.param_groups:
                param_group['lr'] = model.g_lr * 0.1
                if opt.rank == 0:
                    print('===> Current G learning rate: ', param_group['lr'])
            for param_group in model.dis_optimizer.param_groups:
                param_group['lr'] = model.d_lr * 0.1
                if opt.rank == 0:
                    print('===> Current D learning rate: ', param_group['lr'])
            if epoch > 59:
                FM_initial = 0.0

        test_psnr = test(model, test_data_loader, epoch)
        if opt.rank == 0:
            torchlight_write.print_log("PSNR: %f" % test_psnr)
            torchlight_write.print_log('Best PSNR: %f' % best_psnr)
            checkpoint('latest')
            if best_psnr < test_psnr:
                best_psnr = test_psnr
                checkpoint('best')
                torchlight_write.print_log('Best PSNR: %f' % best_psnr)

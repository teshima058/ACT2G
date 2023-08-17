import json
import random
import PIL
import functools
import utils
import progressbar
import numpy as np
import pandas as pd
import os
import argparse
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

# from model import GCDSVAE
from model import GVAE
# from model_textreconst import GCDSVAE
from option import getOptions


opt = getOptions()
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
mse_loss = nn.MSELoss().cuda()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)

weight = torch.Tensor([1, 1, 1, 10, 10, 1, 10, 10]).cuda()

NLL = torch.nn.NLLLoss(ignore_index=0, reduction='sum')

def cos_sim(v1, v2):
    return torch.matmul(v1, v2.T) / (torch.norm(v1) * torch.norm(v2))


def train(epoch, x_gesture, kf_list, model, optimizer, opt):
    model.zero_grad()

    batch_size = x_gesture.size(0)

    x_gesture = x_gesture.type(torch.float32).to(device)
    recon_g, g_mean, g_post, g_logvar = model(x_gesture)

    mse = 0
    for i in range(len(recon_g)):
        mse += F.mse_loss(recon_g[i][:kf_list[i]], x_gesture[i][:kf_list[i]], reduction='sum') 
    # mse = mse / len(recon_g)

    # mse = 0
    # for i in range(len(recon_g)):
    #     kf = opt.frames
    #     for j in range(opt.frames):
    #         if torch.sum(x_gesture[i][j]) == 1:
    #             kf = j
    #             break
    #     mse += F.mse_loss(recon_g[i][:kf], x_gesture[i][:kf], reduction='sum') 

    l_recon_g = mse

    # To avoid posterior collapse 
    if epoch < opt.sche:
        kld_factor = epoch * (1 / opt.sche)
    else:
        kld_factor = 1

    g_mean = g_mean.view((-1, g_mean.shape[-1])) 
    g_logvar = g_logvar.view((-1, g_logvar.shape[-1])) 
    kld_g = -0.5 * torch.sum(1 + g_logvar - torch.pow(g_mean,2) - torch.exp(g_logvar)) * opt.weight_kld * kld_factor

    loss = 0
    loss += l_recon_g
    loss += kld_g

    model.zero_grad()
    loss.backward()
    optimizer.step()

    return [i.data.cpu().numpy() for i in [loss, l_recon_g, kld_g]]


def fix_seed(seed):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

def main(opt):
    # name = 'CDSVAE_Sprite_epoch-{}_bs-{}_rnn_size={}-g_dim={}-f_dim={}-z_dim={}-lr={}' \
    #        '-weight:kl_f={}-kl_z={}-c_aug={}-m_aug={}-{}-sche_{}-{}'.format(
    #            opt.nEpoch, opt.batch_size, opt.rnn_size, opt.g_dim, opt.f_dim, opt.z_dim, opt.lr,
    #            opt.weight_f, opt.weight_z, opt.weight_c_aug, opt.weight_m_aug,
    #            opt.loss_recon, opt.sche, opt.note)
    # name = 'GDSVAE_{}-keyframe_margin={}_onlycont_freezeBERT_matmul'.format(opt.frames, opt.margin)
    name = 'GVAE_allframe_z={}d_kld={}'.format(opt.z_dim, opt.weight_kld)
    opt.log_dir = '%s/%s' % (opt.log_dir, name)
    train_csv_path = opt.log_dir + '/logs_train.csv'
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    fix_seed(opt.seed)

    if opt.model_path != "":
        train_csv_path = os.path.dirname(opt.model_path) + "/logs_train.csv"
        saved_model = torch.load(opt.model_path)
        opt = saved_model['option']
        first_epoch = saved_model['epoch'] + 1
        model = GVAE(opt).cuda()
        model.load_state_dict(saved_model['model'])
        log = pd.read_csv(train_csv_path)
        log[log['epoch']<first_epoch].to_csv(train_csv_path, index=False)
    else:
        # model, optimizer and scheduler
        model = GVAE(opt)
        model = model.cuda()
        opt.optimizer = optim.Adam
        first_epoch = 0

    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch+1)//2, T_mult=1)

    # dataset
    train_data, test_data = utils.load_dataset_GVAE(opt)
    train_loader = DataLoader(train_data,
                              num_workers=0,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    opt.dataset_size = len(train_data)

    epoch_loss_train = Loss()

    # training and testing
    for epoch in range(first_epoch, first_epoch + opt.nEpoch):
        if epoch and scheduler is not None:
            scheduler.step()

        model.train()
        epoch_loss_train.reset()

        opt.epoch_size = len(train_loader)
        progress = progressbar.ProgressBar(max_value=len(train_loader)).start()
        for i, data in enumerate(train_loader):
            progress.update(i+1)
            x_gesture, kf_list = data['gesture'].cuda(), data['kf_list'].cuda()

            losses = train(i, x_gesture, kf_list, model, optimizer, opt)

            losses = {'loss': losses[0], 'g_recon': losses[1], 'g_kld': losses[2]}
            epoch_loss_train.update(losses)
            
        progress.finish()
        utils.clear_progressbar()
        avg_loss = epoch_loss_train.avg()
        lr = optimizer.param_groups[0]['lr']

        # output process
        print('{}\t | '.format(epoch), end='')
        for key in avg_loss.keys():
            print('{}: {:.5f}\t | '.format(key, avg_loss[key]), end='')
        print('{}: {:.5f}'.format('lr', lr))
        
        if os.path.exists(train_csv_path):
            log = pd.read_csv(train_csv_path)
            n = len(log)
            log.at[n, 'epoch'] = epoch
            log.at[n, 'lr'] = lr
            for key in avg_loss.keys():
                log.at[n, key] = avg_loss[key]
            pd.DataFrame(log).to_csv(train_csv_path, index=False)
        else:
            log = {'epoch':[epoch], 'lr': [lr]}
            log.update(avg_loss)
            pd.DataFrame(log).to_csv(train_csv_path, index=False)

        if epoch > first_epoch:
            # Save minimum recon loss model
            if min(log['loss'][first_epoch:]) == avg_loss['loss']:
                recon_min_epoch = epoch
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'option': opt,
                    'epoch': epoch},
                    '%s/model.pth' % (opt.log_dir))
        
    # os.rename("{}/model_recon.pth".format(opt.log_dir), "{}/model_recon_{}.pth".format(opt.log_dir, recon_min_epoch))

def reorder(sequence, opt=None):  # ([128, 8, 64, 64, 3])
    if opt is None or opt.dataset == 'Sprite':
        return sequence.permute(0,1,4,2,3)  # ([128, 8, 3, 64, 64])
    elif opt.dataset == 'Gesture':
        return sequence.permute(0,1,3,2)

class Loss(object):
    def __init__(self):
        self.reset()

    def update(self, losses):
        keys = losses.keys()
        for key in keys:
            if key in self.losses.keys():
                self.losses[key].append(losses[key])
            else:
                self.losses[key] = []

    def reset(self):
        self.losses = {}

    def avg(self):
        avg = {}
        for key in self.losses.keys():
            avg[key] = np.asarray(self.losses[key]).mean() 
        return avg

if __name__ == '__main__':
    main(opt)



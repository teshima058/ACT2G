import json
import random
import progressbar
import numpy as np
import pandas as pd
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

import utils
from model import ACT2G
from option import getOptions


opt = getOptions()
# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)

NLL = torch.nn.NLLLoss(ignore_index=0, reduction='sum')
BCELoss = nn.BCELoss().to(device)
MSELoss = nn.MSELoss().cuda()
CELoss = nn.CrossEntropyLoss().cuda()

def cos_sim(v1, v2):
    return torch.matmul(v1, v2.T) / (torch.norm(v1) * torch.norm(v2))

def gesture_text_contrastive_loss(f1, f2, margin, positive):   # f.shape = [batch_size, feature_size]
    margin = torch.tensor(margin).cuda()
    positive = torch.tensor(positive).cuda()

    dm = torch.cdist(f1, f2)
    sim_pos = torch.sum(1/2 * (torch.mul(positive, dm) ** 2))
    sim_neg = torch.sum(1/2 * torch.max(torch.tensor(0), margin - torch.mul(1-positive, dm)) ** 2)
    loss = (sim_pos + sim_neg) / (f1.shape[0] ** 2)

    return loss


def train(epoch, x_gesture, x_text, y_text, pos, kf_list, model, optimizer, opt, mode="train"):
    if mode == "train":
        model.zero_grad()

    batch_size = x_gesture.size(0)
    x_gesture = x_gesture.type(torch.float32).to(device)

    recon_g, gesture_feature, text_feature, attn, kf_pred = model(x_gesture, x_text)
    # recon_g, gesture_feature, text_feature, attn = model(x_gesture, x_text)
    
    if opt.wo_attn == 1:
        l_bce = torch.tensor(0)
    else:
        l_bce = BCELoss(attn.type(torch.double), y_text) * opt.camma

    mse = 0
    for i in range(len(recon_g)):
        mse += F.mse_loss(recon_g[i][:kf_list[i]], x_gesture[i][:kf_list[i]], reduction='sum') 
    mse /= len(recon_g)

    l_recon_g = mse * opt.alpha

    l_gt_cont = gesture_text_contrastive_loss(text_feature, gesture_feature, opt.margin, pos) * opt.beta

    # l_kf = CELoss(kf_pred, kf_list.to(torch.int64)) * opt.camma
    l_kf = torch.tensor(0)

    loss = 0
    loss += l_recon_g
    loss += l_bce
    loss += l_gt_cont
    # loss += l_kf

    if mode == "train":
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # return [i.data.cpu().numpy() for i in [loss, l_recon_g, l_bce, l_gt_cont]]
    return [i.data.cpu().numpy() for i in [loss, l_recon_g, l_bce, l_gt_cont, l_kf]]

def fix_seed(seed):
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)

def main(opt):
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    fix_seed(opt.seed)

    if opt.model_path != "":
        saved_model = torch.load(opt.model_path)
        logdir = opt.log_dir
        opt = saved_model['option']

        name = 'ACT2G_allframe_α={}_β={}_γ={}_margin={}'.format(opt.alpha, opt.beta, opt.camma, opt.margin, opt.dropout)
        opt.log_dir = '%s/%s' % (logdir, name)
        os.makedirs(opt.log_dir, exist_ok=True)

        train_csv_path = opt.log_dir + '/logs_train.csv'
        # log = pd.read_csv(train_csv_path)
        # first_epoch = saved_model['epoch'] + 1
        # log[log['epoch']<first_epoch].to_csv(train_csv_path, index=False)
        
        first_epoch = 0
        model = ACT2G(opt).cuda()
        model.load_state_dict(saved_model['model'])

        opt.alpha = 1
        opt.beta = 0

        print("Model is loaded.")

    else:
        name = 'ACT2G_allframe_α={}_β={}_γ={}_margin={}'.format(opt.alpha, opt.beta, opt.camma, opt.margin, opt.dropout)
        opt.log_dir = '%s/%s' % (opt.log_dir, name)
        train_csv_path = opt.log_dir + '/logs_train.csv'
        os.makedirs(opt.log_dir, exist_ok=True)
    
        # model, optimizer and scheduler
        model = ACT2G(opt)
        model = model.cuda()
        opt.optimizer = optim.Adam
        first_epoch = 0
    


    optimizer = opt.optimizer(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=2e-4, T_0=(opt.nEpoch+1)//2, T_mult=1)
    scheduler = None

    # dataset
    train_data, test_data, positive = utils.load_dataset(opt)
    train_loader = DataLoader(train_data,
                              num_workers=0,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    opt.dataset_size = len(train_data)

    epoch_loss_train = Loss()

    cost = 100000

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
            x_gesture, x_text, y_text, kf_list = data['gesture'].cuda(), data['text'].cuda(), data['text_label'].cuda(), data['kf_list'].cuda()

            pos = positive[data['index']][:,data['index']]

            # loss, recon, kld = train(i, x_gesture, x_text, model, optimizer, opt)
            losses = train(i, x_gesture, x_text, y_text, pos, kf_list, model, optimizer, opt)

            # losses = {'loss': losses[0], 'l_recon_g': losses[1], 'l_bce': losses[2], 'l_gt_cont': losses[3]}
            losses = {'loss': losses[0], 'l_recon_g': losses[1], 'l_bce': losses[2], 'l_gt_cont': losses[3], 'l_kf': losses[4]}
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

            # # Save minimum kld loss model
            # if cost > avg_loss['cost_g'] + avg_loss['cost_t']:
            #     contrast_min_epoch = epoch
            #     cost = avg_loss['cost_g'] + avg_loss['cost_t']
            #     torch.save({
            #         'model': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'option': opt},
            #         '%s/model_contrast.pth' % (opt.log_dir))

    
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



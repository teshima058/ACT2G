import re
from operator import rshift
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os
import json
import pickle
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import chi2, norm
import copy
from transformers import BertTokenizer

# from model_textreconst import GCDSVAE
from model import GVAE
# from model_prev import GCDSVAE
from plotPose import Plot
from option import getOptions

opt = getOptions()

def reorder(sequence):  # ([128, 8, 64, 64, 3])
    return sequence.permute(0,1,3,2)

# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plotUpperBody2D(pose3d, save_path, fps=25, isRelative=False, frames=None):
    if isRelative:
        new_pose3d = []
        for pose in pose3d:
            new_pose = np.zeros([8, 3])
            new_pose[0] = pose[0]
            new_pose[2] = pose[1]
            new_pose[3] = new_pose[2] + pose[2]
            new_pose[4] = new_pose[3] + pose[3]
            new_pose[5] = pose[4]
            new_pose[6] = new_pose[5] + pose[5]
            new_pose[7] = new_pose[6] + pose[6]
            new_pose3d.append(new_pose)
        pose3d = np.array(new_pose3d)

    if pose3d.shape[1] == 25:
        upper_idx = [2, 20, 4, 5, 6, 8, 9, 10]
        pose3d = pose3d[:, upper_idx]
        pose2d = np.delete(pose3d, 2, 2).reshape([-1, pose3d.shape[1]*2])
    else:
        pose2d = np.delete(pose3d, 2, 2).reshape([-1, pose3d.shape[1]*2])

    if frames is not None:
        pose2d = np.append(pose2d.T, [frames], axis=0).T

    # p = Plot((-0.5, 0.5), (-0.6, 0.4))
    p = Plot((-2, 2), (-2.4, 1.6))
    anim = p.animate(pose2d, 1000/fps)
    p.save(anim, save_path, fps=fps)

class ConfidenceEllipse:
    def __init__(self, data, p=0.95):
        self.data = data
        self.p = p
        self.means = np.mean(data, axis=0)
        self.cov = np.cov(data[:,0], data[:,1])

        lambdas, vecs = np.linalg.eigh(self.cov)
        order = lambdas.argsort()[::-1]
        lambdas, vecs = lambdas[order], vecs[:,order]

        c = np.sqrt(chi2.ppf(self.p, 2))
        self.w, self.h = 2 * c * np.sqrt(lambdas)
        self.theta = np.degrees(np.arctan(
            ((lambdas[0] - lambdas[1])/self.cov[0,1])))
        
    def get_params(self):
        return self.means, self.w, self.h, self.theta

    def get_patch(self, line_color="black", face_color="none", alpha=0):
        el = patches.Ellipse(xy=self.means,
                     width=self.w, height=self.h,
                     angle=self.theta, color=line_color, alpha=alpha)
        el.set_facecolor(face_color)
        return el

def main(opt):
    if opt.model_path != '':
        model_path = opt.model_path
        saved_model = torch.load(model_path)
        opt = saved_model['option']
        opt.batch_size = 256
        opt.model_path = model_path
        gdsvae = GVAE(opt)
        gdsvae.load_state_dict(saved_model['model'], strict=False)
    else:
        raise ValueError('missing checkpoint')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    gdsvae = gdsvae.to(device)
    gdsvae.eval()

    # # save gesture images
    # out =  pickle.load(open(opt.dataset, "rb"))
    # gestures = out['gestures']
    # kf_gestures = out['X_train_gesture']
    # texts = out['remarks']
    # kf_list = out['keyframe_list']
    # save_dir = './images/gestures/stick_3-15/'
    # os.makedirs(save_dir, exist_ok=True)
    # for i in tqdm(range(len(gestures))):
    #     text = re.sub(r'[\\/:*?"<>|]+','',texts[i][:30])
    #     save_path = save_dir + str(i).zfill(4) + '_{}.gif'.format(text)
    #     if os.path.exists(save_path):
    #         continue
    #     frames = np.arange(len(gestures[i]))
    #     plotUpperBody2D(gestures[i], save_path, fps=25, isRelative=False, frames=frames)

    #     # kf = len(kf_list[i])
    #     # frames = np.arange(kf)
    #     # plotUpperBody2D(kf_gestures[i][:kf], save_path, fps=4, isRelative=True, frames=frames)
    # exit()

    # --------- load a dataset ------------------------------------
    train_data, test_data = utils.load_dataset_GVAE(opt)
    train_loader = DataLoader(train_data,
                              num_workers=0,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=0,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)
    
    features = []
    gestures = []
    # --------- training loop ------------------------------------
    save_dir = "./images/reconst/{}/".format(os.path.basename(os.path.dirname(opt.model_path)))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'gesture/', exist_ok=True)

    # isRelative = True
    isRelative = False

    texts, gestures = [], []
    g_features = []
    keyframes = []
    for i, data in enumerate(train_loader):

        x_gesture = data['gesture'].cuda()
        x_gesture = x_gesture.type(torch.float32).to(device)

        # gesture_feature, text_feature, recon_g, recon_t = gdsvae(x_gesture, x_text)
        recon_g, g_mean, g_post, g_logvar = gdsvae(x_gesture)

        g_features.append(g_mean.detach().cpu().numpy()[0])
        # gestures.append(data['gesture'].detach().cpu().numpy()[0])

        kf = 0
        for i in range(opt.frames):
            if torch.sum(x_gesture[0][i]) == 0:
                kf = i
                break
        if kf == 0:
            keyframes.append(opt.frames)
        else:
            keyframes.append(kf)


        for j in range(10):
            save_path = save_dir + "gesture/{}_gesture.gif".format(j)
            if not os.path.exists(save_path):
                x_train = x_gesture[j].detach().cpu().numpy()
                kf = opt.frames
                for k,p in enumerate(x_train):
                    if np.sum(p) == 0:
                        kf = k-1
                        break
                frames = np.arange(kf)
                plotUpperBody2D(x_train[:kf], save_path, isRelative=isRelative, fps=5, frames=frames)

            save_path = save_dir + "gesture/{}_gesture_reconst.gif".format(j)
            if not os.path.exists(save_path):
                reconst = recon_g[j].detach().cpu().numpy()
                plotUpperBody2D(reconst[:kf], save_path, isRelative=isRelative, fps=5, frames=frames)
        # exit()


        # Visualize Latent Space
        mean = g_mean.detach().cpu().numpy()
        std = np.sqrt(np.exp(g_logvar.detach().cpu().numpy()))
        rs = []
        for j in range(opt.batch_size):
            rs.append(np.random.normal(loc=mean[j], scale=std[j], size=[100, opt.z_dim]))
        rs = np.array(rs).reshape(-1, opt.z_dim)
        rs_red = TSNE(n_components=2, random_state=0).fit_transform(rs)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # plot Gesture Circle
        for j in range(opt.batch_size):
            el = ConfidenceEllipse(rs_red[j*100:(j+1)*100], p=0.95)
            mean, _, _, _ = el.get_params()
            ax.add_artist(el.get_patch(face_color='darkorange', alpha=0.1))
            plt.text(mean[0], mean[1], str(j), color='darkorange')
        plt.xlim(-75, 75)
        plt.ylim(-75, 75)
        plt.show()
        exit()
        
    n = len(g_features)
    distmat = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1, n):
            # euclidean distance
            distmat[i][j] = np.sum(np.square(g_features[i] - g_features[j]))
    for i in range(n):
        for j in range(0, i):
            distmat[i][j] = distmat[j][i]

    dic = {"distmat": distmat, "g_features":np.array(g_features)}
    dataset = pickle.load(open(opt.dataset, "rb"))
    g_features = np.array(g_features)
    gesture_ids = dataset['gesture_ids']

    result = {'gestures': dataset["gestures"], 'g_features': g_features, 'gesture_ids':gesture_ids, 'dist_mat': distmat}
    np.save("./data/GVAE_features_keyframe_z=32d_kld=0.03.npy", result)
    print("Saved")
    
    # diff = np.mean((g_features - t_features)**2)
    # print("Deifference between text and gesture features: {}".format(diff))
    # diffs = np.mean((g_features - t_features)**2, axis=1)
    # plt.figure()
    # plt.hist(diffs, bins=10)
    # plt.xlim(0, 4)
    # plt.show()



    latent_features = []
    for i, data in enumerate(train_loader):
        x_gesture, x_text = data['gesture'].cuda(), data['text'].cuda()
        x_gesture = x_gesture.type(torch.float32).to(device)
        x_text = x_text.type(torch.LongTensor).to(device)
        f_mean, f_logvar, f_post, z_mean, z_logvar, z_post, t_mean, t_logvar, t_post, recon_x = gdsvae(x_gesture, x_text)
        for j in range(opt.batch_size):
            latent_features.append(z_post[j].detach().cpu().numpy())
    latent_features = np.array(latent_features)
    mean = np.mean(latent_features, axis=0)
    std = np.std(latent_features, axis=0) * 3

    # Save gesture space figure
    size = 7
    z = torch.zeros([size, size, opt.frames, 2])
    x_tmp = torch.arange(-0.7, -0.1, 0.6/size)
    y_tmp = torch.arange(-0.7, 0.3, 1/size)
    for i in range(size):
        for j in range(size):
            for k in range(opt.frames):
                z[i][j][k][0] = x_tmp[i]
                z[i][j][k][1] = y_tmp[j]
                # z[i][j][k][0] = mean[k][0] - (std[k][0] / size) * (size / 2) + (std[k][0] / size) * i
                # z[i][j][k][1] = mean[k][1] - (std[k][1] / size) * (size / 2) + (std[k][1] / size) * j
    z = z.reshape(-1, opt.frames, 2).cuda()
    recons = gdsvae.decode(z)
    recons = recons.detach().cpu().numpy()

    os.makedirs("{}/tmp/".format(save_dir), exist_ok=True)
    for frame in range(opt.frames):
        fig = plt.figure(figsize=(size*5,size*5))
        plt.suptitle("frame={}".format(frame), fontsize=size*5)
        for i in range(size*size):
            pose3d = recons[i][frame]
            if isRelative:
                new_pose = np.zeros([8, 3])
                new_pose[0] = pose3d[0]
                new_pose[2] = pose3d[1]
                new_pose[3] = new_pose[2] + pose3d[2]
                new_pose[4] = new_pose[3] + pose3d[3]
                new_pose[5] = pose3d[4]
                new_pose[6] = new_pose[5] + pose3d[5]
                new_pose[7] = new_pose[6] + pose3d[6]
                pose3d = new_pose

            ax = fig.add_subplot(size, size, i+1)
            ax.plot([pose3d[0][0], pose3d[1][0]], [pose3d[0][1], pose3d[1][1]], lw=5)
            ax.plot([pose3d[1][0], pose3d[2][0]], [pose3d[1][1], pose3d[2][1]], lw=5)
            ax.plot([pose3d[2][0], pose3d[3][0]], [pose3d[2][1], pose3d[3][1]], lw=5)
            ax.plot([pose3d[3][0], pose3d[4][0]], [pose3d[3][1], pose3d[4][1]], lw=5)
            ax.plot([pose3d[1][0], pose3d[5][0]], [pose3d[1][1], pose3d[5][1]], lw=5)
            ax.plot([pose3d[5][0], pose3d[6][0]], [pose3d[5][1], pose3d[6][1]], lw=5)
            ax.plot([pose3d[6][0], pose3d[7][0]], [pose3d[6][1], pose3d[7][1]], lw=5)
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.6, 0.4)
        plt.title("frame={}".format(frame))
        # fig.savefig("{}/tmp/gesture_{}.png".format(save_dir, frame))
        fig.savefig("{}/tmp/gesture_space_frame={}.png".format(save_dir, frame))

    from PIL import Image
    import glob

    files = sorted(glob.glob("{}/tmp/*.png".format(save_dir)))  
    images = list(map(lambda file : Image.open(file) , files))
    images[0].save('{}/gesture_space_-0.7~-0.1_-0.7~0.3.gif'.format(save_dir) , save_all=True , append_images=images[1:] , duration=400 , loop=0)
    # z_mean = torch.Tensor([mean]).cuda()
    # recons = cdsvae.decode(z_mean)
    # recons = recons.permute(0, 1, 3, 2).detach().cpu().numpy()
    # frames = np.arange(8)
    # plotUpperBody2D(recons[0], each_save_dir + "mean_pose.gif", fps=5, frames=frames)


    os.makedirs("{}/tmp/".format(save_dir), exist_ok=True)
    for frame in range(len(recons)):
        fig = plt.figure(figsize=(size*5,size*5))
        plt.suptitle("frame={}".format(frame), fontsize=size*5)
        for i in range(size*size):
            pose2d = recons[frame][i]
            ax = fig.add_subplot(size, size, i+1)
            ax.plot([pose2d[0][0], pose2d[1][0]], [pose2d[0][1], pose2d[1][1]], lw=5)
            ax.plot([pose2d[1][0], pose2d[2][0]], [pose2d[1][1], pose2d[2][1]], lw=5)
            ax.plot([pose2d[2][0], pose2d[3][0]], [pose2d[2][1], pose2d[3][1]], lw=5)
            ax.plot([pose2d[3][0], pose2d[4][0]], [pose2d[3][1], pose2d[4][1]], lw=5)
            ax.plot([pose2d[1][0], pose2d[5][0]], [pose2d[1][1], pose2d[5][1]], lw=5)
            ax.plot([pose2d[5][0], pose2d[6][0]], [pose2d[5][1], pose2d[6][1]], lw=5)
            ax.plot([pose2d[6][0], pose2d[7][0]], [pose2d[6][1], pose2d[7][1]], lw=5)
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylim(-0.5, 0.7)
        plt.title("frame={}".format(frame))
        fig.savefig("{}/tmp/gesture_{}.png".format(save_dir, frame))
    
    from PIL import Image
    import glob

    files = sorted(glob.glob("{}/tmp/*.png".format(save_dir)))  
    images = list(map(lambda file : Image.open(file) , files))
    images[0].save('{}/gesture_space_-3~0_0.5~0.6.gif'.format(save_dir) , save_all=True , append_images=images[1:] , duration=400 , loop=0)


    print()
    # features = np.array(features)
    # result = {'gestures': gestures, 'features': features, 'gesture_ids': gesture_ids}
    # np.save("./dataset/Gesture/latent_codes_8frame_8joints.npy", result)
    # print("Saved")

            

            


def entropy_Hy(p_yx, eps=1E-16):
    p_y = p_yx.mean(axis=0)
    sum_h = (p_y * np.log(p_y + eps)).sum() * (-1)
    return sum_h

def entropy_Hyx(p, eps=1E-16):
    sum_h = (p * np.log(p + eps)).sum(axis = 1)
    # average over images
    avg_h = np.mean(sum_h) * (-1)
    return avg_h

def inception_score(p_yx,  eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score

def KL_divergence(P, Q, eps=1E-16):
    kl_d = P * (np.log(P + eps) - np.log(Q + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    return avg_kl_d

def print_log(print_string, log=None):
    print("{}".format(print_string))
    if log is not None:
        log = open(log, 'a')
        log.write('{}\n'.format(print_string))
        log.close()

if __name__ == '__main__':
    main(opt)

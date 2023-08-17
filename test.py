import re
import glob
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
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.stats import chi2, norm
import copy
import shutil
from transformers import BertModel, BertConfig, BertTokenizer

from model import ACT2G
# from model_wokf import ACT2G
# from model_20220907 import ACT2G
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

    p = Plot((-0.5, 0.5), (-0.6, 0.4))
    # p = Plot((-0.5, 0.5), (-0.4, 0.6))
    anim = p.animate(pose2d, 1000/fps)
    p.save(anim, save_path, fps=fps)

def plotResult(X, labels, text=None):
    unique_labels = set(labels)
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [cmap[i%len(cmap)] for i in range(len(labels))]
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
    for k, col in zip(unique_labels, colors):
        class_member_mask = labels == k
        xy = X_reduced[class_member_mask]
        if text is not None:
            texts = np.array(text)[class_member_mask]
            for i in range(0, len(xy), 1):
                plt.text(xy[i][0], xy[i][1], texts[i], color=col)
            plt.xlim(-40, 70)
            plt.ylim(-45, 45)
        else:
            plt.scatter(xy[:, 0], xy[:, 1], marker="${}$".format(k), label=k, color=col)
    plt.title("KMeans Clustering")
    plt.show()

def visualizePrediction(tokens, pred, labels=False, save_path=None):
    fig, ax = plt.subplots()
    idx = tokens.index('[PAD]')
    # idx = len(tokens)
    if labels is not None:
        data = np.array([pred[:idx], labels[:idx]]).T
    else:
        data = np.array([pred[:idx]]).T
    sns.heatmap(data=data, cmap='OrRd', annot=True, vmin=0, vmax=1)
    ax.set_xticklabels(['Predicted', 'GT'], rotation=0)
    ax.set_yticks(np.arange(idx)+0.5)
    ax.set_yticklabels(tokens[:idx], rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def interpolateKeypose(keyposes, frame=60):   # [keyframe, joint, xyz]
    keyposes = keyposes.transpose([1, 2, 0])
    new_poses = []
    for pose in keyposes:
        tck, u = interpolate.splprep(pose, k=3)
        new = interpolate.splev(np.linspace(0,1,frame), tck, der=0)
        new_poses.append(new)
    new_poses = np.array(new_poses).transpose([2, 0, 1])
    return new_poses


def main(opt):
    if opt.model_path != '':
        model_path = opt.model_path
        saved_model = torch.load(model_path)
        opt = saved_model['option']
        opt.batch_size = 256
        opt.model_path = model_path
        opt.mode = "test"
        model = ACT2G(opt).cuda()
        model.load_state_dict(saved_model['model'])
    else:
        raise ValueError('missing checkpoint')

    # BERT setting
    options_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(options_name)
    bert_model = BertModel.from_pretrained(options_name)
    bert_model.eval()
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device : ", device)

    model = model.to(device)
    model.eval()

    # --------- load a dataset ------------------------------------
    # train_data, test_data = utils.load_dataset_textreconst(opt)
    train_data, test_data, positive = utils.load_dataset(opt)
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

    # # save gesture images
    # out =  pickle.load(open(opt.dataset, "rb"))
    # gestures = out['gestures']
    # texts = out['remarks']
    # save_dir = './images/Gesture/gestures/stick_1002_key/'
    # os.makedirs(save_dir, exist_ok=True)
    # for i in range(len(gestures)):
    #     text = re.sub(r'[\\/:*?"<>|]+','',texts[i][:30])
    #     save_path = save_dir + str(i).zfill(4) + '_{}.gif'.format(text)
    #     if os.path.exists(save_path):
    #         continue
    #     frames = np.arange(len(gestures[i]))
    #     plotUpperBody2D(gestures[i], save_path, fps=4, isRelative=False, frames=frames)
    # exit()
    
    # --------- training loop ------------------------------------
    # save_dir = "./images/Gesture/reconst_train_{}keyframe_1000ges_r-1000_f=256dim_linear/".format(opt.frames)
    save_dir = "./images/reconst/{}/".format(os.path.basename(os.path.dirname(opt.model_path)))

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'gesture/', exist_ok=True)
    os.makedirs(save_dir+'gesture_allframe/', exist_ok=True)
    os.makedirs(save_dir+'gesture_interp/', exist_ok=True)
    os.makedirs(save_dir+'attention/', exist_ok=True)
    os.makedirs(save_dir+'attention_norm/', exist_ok=True)

    isRelative = True
    # isRelative = False

    kf_gestures, recon_gestures, g_features, t_features, attns, kfs, kf_preds = [], [], [], [], [], [], []
    keyframes = []
    for i, data in enumerate(train_loader):
        x_gesture, x_text, y_text, kf_list = data['gesture'].cuda(), data['text'].cuda(), data['text_label'].cuda(), data['kf_list'].cuda()
        x_gesture = x_gesture.type(torch.float32).to(device)
        if opt.batch_size != 1:
            pos = positive[data['index']][:,data['index']]

        recon_g, gesture_feature, text_feature, attn, kf_pred = model(x_gesture, x_text)
        # recon_g, gesture_feature, text_feature, attn = model(x_gesture, x_text)

        # kf_pred = torch.argmax(kf_pred, dim=1).detach().cpu().numpy()
        g_features.append(gesture_feature.detach().cpu().numpy()[0])
        t_features.append(text_feature.detach().cpu().numpy()[0])
        kf_gestures.append(data['gesture'].detach().cpu().numpy()[0])
        recon_gestures.append(recon_g.detach().cpu().numpy()[0])
        attns.append(attn.detach().cpu().numpy()[0])
        kfs.append(kf_list.detach().cpu().numpy()[0])
        # kf_preds.append(kf_pred[0])

        kf_list = kf_list.detach().cpu().numpy()

        # df = pd.DataFrame({"GT": kf_list, "Pred": kf_pred})
        # df.to_excel(save_dir+"keyframe_pred.xlsx", index=False)

        ############   Reconst Gestures   ############
        sep = 1 if opt.frames > 30 else 0
        fps = 25 if opt.frames > 30 else 5
        for j in range(10):
            save_path = save_dir + "gesture/{}_gesture.gif".format(j)
            if not os.path.exists(save_path):
                x_train = x_gesture[j].detach().cpu().numpy()
                frames = np.arange(kf_list[j])
                plotUpperBody2D(x_train[:kf_list[j]], save_path, isRelative=isRelative, fps=fps, frames=frames)

            save_path = save_dir + "gesture_allframe/{}_gesture.gif".format(j)
            if not os.path.exists(save_path):
                reconst = recon_g[j].detach().cpu().numpy()
                frames = np.arange(opt.frames)
                plotUpperBody2D(reconst, save_path, isRelative=isRelative, fps=fps, frames=frames)

            save_path = save_dir + "gesture/{}_gesture_reconst.gif".format(j)
            if not os.path.exists(save_path):
                reconst = recon_g[j].detach().cpu().numpy()
                frames = np.arange(kf_list[j])
                plotUpperBody2D(reconst[:kf_list[j]], save_path, isRelative=isRelative, fps=fps, frames=frames)

            # save_path = save_dir + "gesture_interp/{}_gesture_interp.gif".format(j)
            # if not os.path.exists(save_path):
            #     frame_num = kf_list[j] * 12
            #     keyposes = recon_g[j].detach().cpu().numpy()[:kf_pred[j]]
            #     new_poses = interpolateKeypose(keyposes, frame=frame_num)
            #     plotUpperBody2D(new_poses, save_path, isRelative=isRelative, fps=25, frames= np.arange(frame_num))

        # ############   Visualize Multi-model 2D Space   ############
        print("visualizing multimodal space")
        text_f = text_feature.detach().cpu().numpy()
        gest_f = gesture_feature.detach().cpu().numpy()
        all = np.concatenate([gest_f, text_f])
        # all_red = TSNE(n_components=2, random_state=0).fit_transform(all)
        all_red = PCA(n_components=2, random_state=0).fit_transform(all)
        gest_red = all_red[:opt.batch_size]
        text_red = all_red[opt.batch_size:]
        plt.figure()
        for i in range(opt.batch_size):
            # plt.text(gest_red[i][0], gest_red[i][1], str(i), color='darkorange')
            # plt.text(text_red[i][0], text_red[i][1], str(i), color='blue')
            plt.plot(gest_red[i][0], gest_red[i][1], marker="o", color='darkorange')
            plt.plot(text_red[i][0], text_red[i][1], marker="o", color='blue')
        for i in range(opt.batch_size):
            for j in range(i+1, opt.batch_size):
                if pos[i][j] == 1:
                    plt.plot([gest_red[i][0], gest_red[j][0]], [gest_red[i][1], gest_red[j][1]], 
                        linestyle="dashed", color="r", marker="None", lw=1)
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.show()
        exit()

        ############   Visualize Attention Weight   ############
        a = attn.detach().cpu().numpy()
        l = y_text.detach().cpu().numpy()
        a_n = np.zeros([len(a[i]), len(a[i])])
        for i in range(len(a)):
            for j in range(len(a[i])):
                a_n[i][j] = a[i][j] / sum(a[i])

        for j in range(opt.batch_size):
            tokens = tokenizer.convert_ids_to_tokens(x_text[j])

            save_path = save_dir+"attention/{}.png".format(j)
            visualizePrediction(tokens, a[j], l[j], save_path)

            save_path = save_dir+"attention_norm/{}.png".format(j)
            visualizePrediction(tokens, a_n[j], l[j], save_path)

        exit()


    g_features = np.array(g_features)
    t_features = np.array(t_features)
    kfs = np.array(kfs)
    kf_preds = np.array(kf_preds)

    print("Average of Keyframe Difference Pred - GT = ", np.mean(kf_preds - kfs))

    plt.figure()
    plt.hist([kf_preds, kfs], label=['Prediction', 'GT'])
    plt.legend()
    plt.show()

    idx_list = np.where(pos[13] == 1)[0]
    t = text_feature[idx_list].detach().cpu().numpy()
    g = gesture_feature[idx_list].detach().cpu().numpy()
    dm = np.zeros([len(t), len(g)])
    for i in range(len(t)):
        for j in range(len(g)):
            dm[i][j] = np.linalg.norm(t[i] - g[j], ord=2)
    for i in range(len(dm)):
        print(np.argmin(dm[i]))

    ############   Output Pair Difference  ############
    diff = np.mean((g_features - t_features)**2)
    print("Difference between text and gesture features: {}".format(diff))
    diffs = np.mean((g_features - t_features)**2, axis=1)
    plt.figure()
    plt.hist(diffs, bins=10)
    plt.xlim(0, 4)
    plt.savefig(save_dir + "diff_{}.png".format(diff))

    dataset = pickle.load(open(opt.dataset, "rb"))
    gestures = dataset["gestures"]
    if opt.frames > 100:
        keyframes = [len(g) for g in dataset["gestures"]]
    else:
        keyframes = [len(g) for g in dataset["keyframe_list"]]
    texts = dataset['remarks']
    result = {'gestures': gestures, 'texts': texts, 'g_features': g_features, 't_features': t_features, 'keyframes': keyframes}
    np.save("./data/features_keyframe_z={}d_g={}_distmat=vae_thresh={}.npy".format(opt.z_dim, opt.weight_g_cont, opt.distance_thresh), result)
    print("Saved")
    


            
if __name__ == '__main__':
    main(opt)

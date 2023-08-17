import os
import torch
from torch.utils.data import DataLoader
import pickle
import utils
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import interpolate
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.cluster import KMeans

from model import ACT2G
# from model_wokf import ACT2G
from plotPose import Plot
from option import getOptions
from clustering_kmeans import plotResult, elbowMethod

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
        opt.batch_size = 1
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
    
    save_dir = "./images/clustering/reclustering_{}".format(os.path.basename(os.path.dirname(opt.model_path)))
    save_path = "./data/library_{}.npy".format(os.path.basename(os.path.dirname(opt.model_path)))
    os.makedirs(save_dir, exist_ok=True)

    isRelative = True

    kf_gestures, recon_gestures, g_features, t_features, attns= [], [], [], [], []
    keyframes = []
    for i, data in enumerate(train_loader):
        x_gesture, x_text, y_text, kf_list = data['gesture'].cuda(), data['text'].cuda(), data['text_label'].cuda(), data['kf_list'].cuda()
        x_gesture = x_gesture.type(torch.float32).to(device)
        if opt.batch_size != 1:
            pos = positive[data['index']][:,data['index']]

        recon_g, gesture_feature, text_feature, attn, kf_pred = model(x_gesture, x_text)
        # recon_g, gesture_feature, text_feature, attn = model(x_gesture, x_text)

        g_features.append(gesture_feature.detach().cpu().numpy()[0])
        t_features.append(text_feature.detach().cpu().numpy()[0])
        kf_gestures.append(data['gesture'].detach().cpu().numpy()[0])
        recon_gestures.append(recon_g.detach().cpu().numpy()[0])
        attns.append(attn.detach().cpu().numpy()[0])
        kf_list = kf_list.detach().cpu().numpy()

    g_features = np.array(g_features)
    t_features = np.array(t_features)

    data = pickle.load(open(opt.dataset, "rb"))


    elbowMethod(t_features, iter_num=60)

    n_clusters = 30
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(t_features)
    t_labels = kmeans.labels_
    plotResult(t_features, t_labels)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(g_features)
    g_labels = kmeans.labels_
    plotResult(g_features, g_labels)


    # extract gesture speed
    gesture_speeds = []
    for gesture in data['gestures']:
        l_hand = gesture[:,6, :]
        r_hand = gesture[:,10, :]
        l_speed = np.linalg.norm(l_hand[:-1] - l_hand[1:], ord=2)
        r_speed = np.linalg.norm(r_hand[:-1] - r_hand[1:], ord=2)
        gesture_speeds.append((l_speed + r_speed) / 2)
    gesture_speeds = np.array(gesture_speeds)

    lib = {}
    lib['t_features'] = t_features
    lib['g_features'] = g_features
    lib['gestures'] = data['gestures']
    lib['texts'] = data['texts']
    lib['gesture_ids'] = data['gesture_ids']
    lib['g_labels'] = g_labels
    lib['g_speeds'] = gesture_speeds
    lib['model'] = saved_model

    np.save(save_path, lib)
    print('Saved to ', save_path)
    


            
if __name__ == '__main__':
    main(opt)

import os
import re
import glob
import json
import shutil
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from PIL import Image
from plotPose import Plot
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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
    anim = p.animate(pose2d, 1000/fps)
    p.save(anim, save_path, fps=fps)


def findIndex(lst, value):
    return [i for i, x in enumerate(lst) if x == value]
    

def createGIF(gesture, gesture_ids):
    for i,poses in tqdm(enumerate(gesture)):
        save_path = "./images/Gesture/GIF/{}.gif".format(gesture_ids[i])
        if os.path.exists(save_path):
            continue
        plotUpperBody2D(poses, save_path, fps=25)

def plotResult(X, labels, text=None):
    unique_labels = set(labels)
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [cmap[i%len(cmap)] for i in range(len(labels))]
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
    plt.figure()
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

def elbowMethod(features, iter_num=30):
    distortions = []
    for i in tqdm(range(1, iter_num+1)):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(features)
        distortions.append(kmeans.inertia_)

        # kmedoids = KMedoids(n_cluster=i, random_state=2)
        # labels = kmedoids.fit_predict(dist_mat)
        # distortions.append(kmedoids.calc_sse())

    plt.plot(range(1, iter_num+1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.show()

if __name__  == "__main__":
    n_clusters = 40
    isRelative = True
    gesture_image_dir = './images/gestures/stick_3-15/'
    dataset_path = "./dataset/data_3-15keyframe_8joints_textlabel.pkl"
    data_path = "./data/GVAE_features_keyframe_z=32d_kld=0.03.npy"
    data_path = './data/GVAE_features_keyframe_z=32d_kld=0.1_nopad.npy'
    out = np.load(data_path, allow_pickle=True).item()
    data = pickle.load(open(dataset_path, "rb"))
    g_features = out['g_features']
    # t_features = data['t_features']
    gestures = out['gestures']
    texts = data['texts']
    keyframes = data['keyframe_list']

    save_dir = "./images/clustering/{}_k={}/".format(os.path.basename(data_path)[:-4], n_clusters)
    os.makedirs(save_dir, exist_ok=True)

    # diff = np.mean(np.linalg.norm(g_features - t_features, ord=2))
    # print("Deifference between text and gesture features: {}".format(diff))

    # features = np.reshape(features, [features.shape[0], features.shape[1] * features.shape[2]])

    # features = gestures.reshape(gestures.shape[0], -1)

    # createGIF(gesture, gesture_ids)
    # exit()

    # elbowMethod(g_features, iter_num=70)
    # exit()


    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(g_features)
    g_labels = kmeans.labels_

    # Show result
    plotResult(g_features, g_labels)

    # with open(save_dir + 'result.txt', 'w') as f:
    #     f.write("Pseudo F: {}\n".format(calinski_harabasz_score(g_features, g_labels)))
    #     f.write("Silhouette Score: {}\n".format(silhouette_score(features, g_labels)))
    #     f.write("Davies Bouldin: {}\n".format(davies_bouldin_score(features, g_labels)))


    if not 'positive' in out.keys():
        positive = np.zeros([len(g_labels), len(g_labels)])
        for i in range(len(g_labels)):
            for j in range(len(g_labels)):
                if g_labels[i] == g_labels[j]:
                    positive[i][j] = 1
                else:
                    positive[i][j] = 0
        out['positive'] = positive
        np.save(data_path, out)
        print("Saved")


    # Save gesture images
    keyframes = [len(l) for l in data['keyframe_list']]

    # for i,label in tqdm(enumerate(g_labels)):
    #     # if i % 40 != 0:
    #     #     continue

    #     each_save_dir = save_dir + "{}/".format(label)
    #     os.makedirs(each_save_dir, exist_ok=True)

    #     if len(glob.glob(gesture_image_dir + str(i).zfill(4) + '*.gif')) == 0:
    #         continue

    #     kf = keyframes[i]
    #     gesture_path = glob.glob(gesture_image_dir + str(i).zfill(4) + '*.gif')[0]
    #     distance_from_center = np.sum(np.square(kmeans.cluster_centers_[label] - g_features[i]))
    #     # save_path = each_save_dir + '{}_kf{}_{}.gif'.format(i, kf, re.sub(r'[\\/:*?"<>|]+','',texts[i][:-1][:30]))
    #     save_path = each_save_dir + '{:.3f}_kf{}_{}.gif'.format(distance_from_center, kf, re.sub(r'[\\/:*?"<>|]+','',texts[i][:30]))
        
    #     if os.path.exists(save_path):
    #         continue

    #     shutil.copyfile(gesture_path, save_path)

    text_len = [len(t.split()) for t in texts]
    keyframes = [len(l) for l in data['keyframe_list']]
    frames = [len(g) for g in data['gestures']]
    df = pd.DataFrame({'id': np.arange(len(g_labels)), 'label': g_labels, 'text': texts, 'text length': text_len, 'keyframes': keyframes, 'frames': frames})
    df.to_excel(save_dir + "./gesture_clustering_result.xlsx", index=False)


    print()
    # gesture_list    = geslib.item().get('gesture_list')         # [cluster(34), n_gestures, frame, joint, xyz]
    # laban_list      = geslib.item().get('laban_list')           # [cluster(34), n_gestures, laban_dic]
    # distance_list   = geslib.item().get('distance_list')        # [cluster(34), n_gestured, n_gestures]
    # remark_list     = geslib.item().get('remark_list')          # [n_remarks (3262), text]
    # index2cluster   = geslib.item().get('index_to_cluster')     # [n_data (3262)]
    # embvecs         = geslib.item().get('emb_vectors')          # [n_data (3262), 768]
    # cluster2index   = geslib.item().get('cluster_to_index')     # [cluster(34), n_gestures]



    # # Text Clustering after Gesture Clustering
    # new_save_dir = "./images/Gesture/{}_k={}_text_clustering/".format(os.path.basename(data_path)[:-4], n_clusters)
    # os.makedirs(new_save_dir, exist_ok=True)
    # data_path = "./dataset/data_5-12keyframe_8joints_1000gestures_reannot_BERT.pkl"
    # dataset = pickle.load(open(data_path, "rb"))
    # t_features = dataset['X_train_text']
    # new_labels = np.zeros(len(g_features))
    # label_count = 0
    # for i in range(n_clusters):
    #     idx = findIndex(g_labels, i)
    #     text_features = t_features[idx]
    #     k = int(len(idx) / 10) + 1
    #     t_labels = KMeans(n_clusters=k, random_state=0).fit_predict(text_features)

    #     for j,label in tqdm(enumerate(t_labels)):
    #         new_labels[idx[j]] = label_count + t_labels[j]
    #         each_save_dir = new_save_dir + "{}_{}/".format(i, int(new_labels[idx[j]]))
    #         os.makedirs(each_save_dir, exist_ok=True)

    #         if len(glob.glob(gesture_image_dir + str(i).zfill(4) + '*.gif')) == 0:
    #             continue


    #         gesture_path = glob.glob(gesture_image_dir + str(idx[j]).zfill(4) + '*.gif')[0]
    #         save_path = each_save_dir + '{}_{}.gif'.format(str(idx[j]).zfill(4), re.sub(r'[\\/:*?"<>|]+','', texts[idx[j]][:30]))
            
    #         if os.path.exists(save_path):
    #             continue

    #         shutil.copyfile(gesture_path, save_path)
        
    #     label_count += max(t_labels) + 1
    # g_labels = new_labels

    # Save Gesture Library
    save_geslib_path = "./data/gesture_library_{}_k={}.npy".format(os.path.basename(data_path)[:-4], n_clusters)
    dataset_path = "./dataset/data_5-12keyframe_8joints_1000gestures_reannot_BERT_distmat=vae.pkl"
    dataset = pickle.load(open(dataset_path, "rb"))

    distance_list = []
    remark_list = dataset['remarks']
    index2cluster = g_labels
    embvecs = dataset['X_train_text']

    cluster2index = {}
    for i in range(len(g_labels)):
        n = g_labels[i]
        if n in cluster2index.keys():
            cluster2index[n].append(i)
        else:
            cluster2index[n] = [i]

    gesture_list = []
    for l in range(n_clusters):
        gesture_list.append(dataset['gestures'][cluster2index[l]])

    data_dir = "Z:/Human/teshima/TED_videos/segmented_by_gesture"
    labans = []
    for i in range(len(dataset['gesture_ids'])):
        gesture_id = dataset['gesture_ids'][i]
        video_id = gesture_id[:11]
        laban_json = data_dir + '/' + video_id + '/' + gesture_id + '/' + gesture_id + '.json'
        with open(laban_json, 'r') as f:
            laban = json.load(f)
        labans.append(laban)
    labans = np.array(labans)
    laban_list = []
    for l in range(n_clusters):
        laban_list.append(labans[cluster2index[l]])

    gesture_library = { 'gesture_list':gesture_list, 
                        'laban_list':laban_list, 
                        'distance_list':distance_list, 
                        'remark_list':remark_list, 
                        'index_to_cluster':index2cluster, 
                        'cluster_to_index':cluster2index, 
                        'emb_vectors':embvecs,
                        'gesture_feature': g_features,
                        'text_feature': t_features}
                        
    np.save(save_geslib_path, gesture_library)
    print('Saved Gesture Library.')


    # text_len = [len(t.split()) for t in texts]
    # df = pd.DataFrame({'id': np.arange(len(g_labels)), 'label': g_labels, 'new label': new_labels, 'text': texts, 'text length': text_len, 'keyframes': keyframes})
    # df.to_excel(new_save_dir + "./gesture_clustering_result.xlsx", index=False)


    #  Text Clusterint_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(t_features)
    # df = pd.DataFrame({'id': np.arange(len(t_labels)), 'label': t_labels, 'text': texts, 'text length': text_len, 'keyframes': keyframes})
    # df.to_csv(save_dir + "./text_clustering_result.csv", index=False)

    # plotResult(t_features, t_labels, text=texts)
    plotResult(g_features, g_labels, text=texts)

    print()
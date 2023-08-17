import os
import glob
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_kinect_csv(csv_file):
    kinect_3d_pose = pd.read_csv(csv_file, header=None)
    pose3d = []
    for i in range(len(kinect_3d_pose)):
        frame_pose = kinect_3d_pose.iloc[i]
        pose = []
        for j in range(len(frame_pose)):
            if j == 0:
                tmp = []
            elif j % 4 == 0:
                pose.append(tmp)
                tmp = []
            else:
                tmp.append(frame_pose[j])
        pose3d.append(pose)
    pose3d = np.array(pose3d)
    return pose3d

if __name__ == "__main__":
    data_dir = "C:/Users/b19.teshima/Documents/Gesture/Evaluations/Training_data/kinect_csv"
    laban_dir = "C:/Users/b19.teshima/Documents/Gesture/Evaluations/Training_data/laban"
    save_path = "./dataset/GVAE_TrinityDataset.pkl"
    valid_joints = [3, 20, 4, 5, 6, 8, 9, 10]
    max_frame = 3000
    max_keyframe = 120

    pad = np.zeros([1,len(valid_joints),3])

    file_list = glob.glob(data_dir + '/*.csv')
    motions, lengths, keyframes = [], [], []
    print('Loading csv files...')
    for csv in tqdm(file_list):
        motion = load_kinect_csv(csv)
        lengths.append(len(motion))

        # Normalize shoulder length to 1 and Spine shoulder to (0, 0, 0)
        normalized_gesture = np.zeros(motion.shape)
        for f in range(len(motion)):
            shoulder_length = np.linalg.norm(motion[f][4] - motion[f][8], ord=2)
            tmp = motion[f] / shoulder_length
            tmp -= tmp[20]
            normalized_gesture[f] = tmp
        motions.append(normalized_gesture[:,valid_joints,:])

        laban_path = laban_dir + '/' + os.path.basename(csv)[:-4] + '.json'
        with open(laban_path, 'r') as f:
            laban = json.load(f)
            laban = laban[list(laban.keys())[0]]
        keyframe = []
        for key in list(laban.keys()):
            keyframe.append(int(int(laban[key]['start time'][0])/1000 * 25))
        keyframes.append(keyframe)

    motions = np.array(motions)
    keyframes = np.array(keyframes)
    
    aligned_motions, valid_motions = [], []
    for i in range(len(motions)):
        if len(motions[i]) > max_frame:
            continue
        valid_motions.append(motions[i])
        while(len(motions[i]) < max_frame):
            motions[i] = np.concatenate([motions[i], pad])
        aligned_motions.append(motions[i])
    aligned_motions = np.array(aligned_motions)
    valid_motions = np.array(valid_motions)

    aligned_key_motions, valid_key_motions, valid_keyframes = [], [], []
    for i in range(len(motions)):
        key_motion = motions[i][keyframes[i]]
        if len(key_motion) > max_keyframe:
            continue
        valid_key_motions.append(key_motion)
        valid_keyframes.append(np.array(keyframes[i]))
        while(len(key_motion) < max_keyframe):
            key_motion = np.concatenate([key_motion, pad])
        aligned_key_motions.append(key_motion)
    aligned_key_motions = np.array(aligned_key_motions)
    valid_key_motions = np.array(valid_key_motions)  
    valid_keyframes = np.array(valid_keyframes)  

    data = {}
    data['valid_gestures'] = aligned_motions
    data['gestures'] = valid_motions
    data['X_train_gesture'] = aligned_key_motions
    data['key_gestures'] = valid_key_motions
    data['keyframe_list'] = valid_keyframes

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print("Done")

    print()
    


    




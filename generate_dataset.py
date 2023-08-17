import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertConfig


def toRelativePosition(upper_gestures, max_frame, frame_list):
    relative_gestures = []
    for i in range(len(upper_gestures)):
        g = upper_gestures[i]
        gesture = []
        for frame in range(len(g)):
            if frame == frame_list[i]:
                gesture.append(np.ones([7,3]))
                for j in range(max_frame - frame_list[i] - 1):
                    gesture.append(np.zeros([7,3]))
                break
            pose = []
            pose.append(g[frame][0] - g[frame][1])
            pose.append(g[frame][2] - g[frame][1])
            pose.append(g[frame][3] - g[frame][2])
            pose.append(g[frame][4] - g[frame][3])
            pose.append(g[frame][5] - g[frame][1])
            pose.append(g[frame][6] - g[frame][5])
            pose.append(g[frame][7] - g[frame][6])
            gesture.append(pose)
        relative_gestures.append(np.array(gesture))
    return np.array(relative_gestures)



if __name__ == '__main__':
    library_path = "./data/imagistic_gestures.npy"
    bert_model = 'bert-base-uncased'
    # valid_keyframe_num = [5,6,7,8,9,10,11,12]
    valid_keyframe_num = np.arange(3, 16)
    valid_joints = [3, 20, 4, 5, 6, 8, 9, 10]
    max_frame = 170     # max([len(g) for g in data['gestures']])

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    library = np.load(library_path, allow_pickle=True).item()
    gestures = library['gesture']
    labans = library['laban']
    texts = library['text']
    remarks = library['remark']
    gesture_ids = library['gesture_id']

    max_keyframe = max(valid_keyframe_num)
    end = np.zeros([1, len(valid_joints), 3]) + 1
    pad = np.zeros([1, len(valid_joints), 3])

    kf_upper_gestures, upper_gestures = [], []
    keyframe_list = []
    valid_gestures, valid_gesture_ids, valid_texts, valid_remarks = [], [], [], []
    for i in tqdm(range(len(gestures))):
        laban = labans[i][list(labans[i].keys())[0]]
        keyframes = []
        for key in list(laban.keys()):
            keyframes.append(int(int(laban[key]['start time'][0])/1000 * 25))
        if len(keyframes) in valid_keyframe_num:
            kf_upper_gesture = gestures[i][keyframes][:,valid_joints,:]
            upper_gesture = gestures[i][:,valid_joints,:]

            if len(keyframes) < max_keyframe:
                kf_upper_gesture = np.vstack([kf_upper_gesture, end])       # add end pose
            
            while len(kf_upper_gesture) < max_keyframe:
                kf_upper_gesture = np.vstack([kf_upper_gesture, pad])       # add padding pose

            # if len(keyframes) < max_keyframe:
            #     upper_gesture = np.vstack([upper_gesture, end])       # add end pose
            
            # while len(upper_gesture) < max_frame:
            #     upper_gesture = np.vstack([upper_gesture, pad])       # add padding pose

            kf_upper_gestures.append(kf_upper_gesture)
            # upper_gestures.append(upper_gesture)
            keyframe_list.append(keyframes)
            valid_gestures.append(gestures[i])
            valid_gesture_ids.append(gesture_ids[i])
            valid_texts.append(texts[i])
            valid_remarks.append(remarks[i])
    
    x_text = []
    for i in range(len(valid_remarks)):
        x_text.append(tokenizer(valid_remarks[i])['input_ids'])

    kf_list = [len(g) for g in keyframe_list]
    frame_list = [len(g) for g in valid_gestures]
    relative_kf_gestures = toRelativePosition(kf_upper_gestures, max_keyframe, kf_list)
    # relative_gestures = toRelativePosition(upper_gestures, max_frame, frame_list)

    data = {}
    data['X_train_gesture'] = np.array(relative_kf_gestures)
    data['X_test_gesture'] = np.array(relative_kf_gestures)
    data['X_train_text'] = np.array(x_text)
    data['X_test_text'] = np.array(x_text)
    data['keyframe_list'] = np.array(keyframe_list)
    data['gestures'] = np.array(valid_gestures)
    # data['valid_gestures'] = np.array(relative_gestures)
    data['gesture_ids'] = np.array(valid_gesture_ids)
    data['texts'] = np.array(valid_texts)
    data['remarks'] = np.array(valid_remarks)
    with open("./dataset/data_3-15keyframe_8joints.pkl", 'wb') as f:
        pickle.dump(data, f)
    
    print("Done")
    
    






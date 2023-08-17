import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import torch

from transformers import BertModel, BertConfig, BertTokenizer
import nltk


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


def add_distance_matrix():
    data_path = "./dataset/data_3-15keyframe_8joints_textlabel.pkl"
    save_path = "./dataset/data_3-15keyframe_8joints_distmat=vae.pkl"
    distmat_path = "./data/GVAE_features_keyframe_z=32d_kld=0.1_nopad.npy"
    dataset = pickle.load(open(data_path, "rb"))
    distmat_data = np.load(distmat_path, allow_pickle=True).item()

    # --- for laban distmat ---
    # d = np.load('./data/distmat_1000reannot_laban.npy', allow_pickle=True).item()
    # dm = d['dist_mat']
    # arranged_dm = np.zeros(list(dm.shape))
    # g_ids = [os.path.basename(j)[:-5] for j in d['json_list']]
    # for i,g_id1 in enumerate(dataset['gesture_ids']):
    #     for j,g_id2 in enumerate(dataset['gesture_ids']):
    #         idx1 = g_ids.index(g_id1)
    #         idx2 = g_ids.index(g_id2)
    #         arranged_dm[i][j] = dm[idx1][idx2]
    # dataset['dist_mat'] = arranged_dm

    # --- for vae distmat ---
    dataset['dist_mat'] = distmat_data['dist_mat']
    dataset['gesture_feature'] = distmat_data['g_features']
    dataset['positive'] = distmat_data['positive']

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

def createNewData():
    data_path = "./dataset/data_3-15keyframe_8joints.pkl"
    savedata_path = "./dataset/data_3-15keyframe_8joints_textlabel.pkl"
    bert_model = 'bert-base-uncased'

    dataset = pickle.load(open(data_path, "rb"))
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    text_labels = []
    for i in tqdm(range(len(dataset['X_train_gesture']))):

        text_token = tokenizer(str(dataset["texts"][i]).lower())['input_ids']
        remark_token = tokenizer(str(dataset["remarks"][i]).lower())['input_ids'][1:-1]

        label = []
        for j in range(len(text_token)):
            if j == 0:  # CLS
                label.append(0)
            elif j == len(text_token)-1:    # SEP
                label.append(0)
            else:
                found = False
                for k in range(len(remark_token)):
                    if remark_token[k] == text_token[j]:
                        label.append(1)
                        remark_token = remark_token[1:]
                        found = True
                        break
                if not found:
                    label.append(0)
        text_labels.append(np.array(label))

    dataset['text_labels'] = np.array(text_labels)

    with open(savedata_path, 'wb') as f:
        pickle.dump(dataset, f)
    print()

def createTestData():
    library_path = "./data/imagistic_gestures.npy"
    # data_path = "./dataset/data_5-12keyframe_1000gestures_BERT_distmat=vae.pkl"
    data_path = "./dataset/data_3-15keyframe_8joints.pkl"
    # savedata_path = "./dataset/testdata_5-12keyframe.pkl"
    savedata_path = "./dataset/data_3-15keyframe_8joints_textlabel.pkl"
    max_keyframe = 15   # max([len(g) for g in data['keyframe_list']])
    valid_joints = [3, 20, 4, 5, 6, 8, 9, 10]
    # max_frame = 240     # max([len(g) for g in data['gestures']])
    bert_model = 'bert-base-uncased'

    library = np.load(library_path, allow_pickle=True).item()
    dataset = pickle.load(open(data_path, "rb"))
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    gestures = library['gesture']
    labans = library['laban']
    texts = library['text']
    remarks = library['remark']
    gesture_ids = library['gesture_id']

    end = np.zeros([1, len(valid_joints), 3]) + 1
    pad = np.zeros([1, len(valid_joints), 3])

    kf_upper_gestures, upper_gestures = [], []
    keyframe_list, text_labels = [], []
    valid_gestures, valid_gesture_ids, valid_texts, valid_remarks = [], [], [], []
    for i in tqdm(range(len(gestures))):

        gesture_id = gesture_ids[i]
        if gesture_id in dataset['gesture_ids']:
            continue

        laban = labans[i][list(labans[i].keys())[0]]
        keyframes = []
        for key in list(laban.keys()):
            keyframes.append(int(int(laban[key]['start time'][0])/1000 * 25))
            
        upper_gesture = gestures[i][:,valid_joints,:]

        if len(keyframes) < max_keyframe:
            kf_upper_gesture = np.vstack([kf_upper_gesture, end])       # add end pose
        
        while len(kf_upper_gesture) < max_keyframe:
            kf_upper_gesture = np.vstack([kf_upper_gesture, pad])       # add padding pose

        # if len(keyframes) < max_keyframe:
        #     upper_gesture = np.vstack([upper_gesture, end])       # add end pose
        
        # while len(upper_gesture) < max_frame:
        #     upper_gesture = np.vstack([upper_gesture, pad])       # add padding pose

        text_token = tokenizer(str(texts[i]).lower())['input_ids']
        remark_token = tokenizer(str(remarks[i]).lower())['input_ids'][1:-1]

        label = []
        for j in range(len(text_token)):
            if j == 0:  # CLS
                label.append(0)
            elif j == len(text_token)-1:    # SEP
                label.append(0)
            else:
                found = False
                for k in range(len(remark_token)):
                    if remark_token[k] == text_token[j]:
                        label.append(1)
                        remark_token = remark_token[1:]
                        found = True
                        break
                if not found:
                    label.append(0)
        text_labels.append(np.array(label))

        kf_upper_gestures.append(kf_upper_gesture)
        upper_gestures.append(upper_gesture)
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
    data['text_labels'] = np.array(text_labels)

    with open(savedata_path, 'wb') as f:
        pickle.dump(data, f)
    print()


def fixPadding():
    data_path = "./dataset/data_5-12keyframe_1000gestures_BERT_distmat=vae.pkl"
    savedata_path = "./dataset/data_5-12keyframe_1000gestures_BERT_distmat=vae_pad=mean.pkl"
    max_keyframe = 12
    dataset = pickle.load(open(data_path, "rb"))

    gestures = dataset['X_train_gesture']
    keyframes = [len(k) for k in dataset['keyframe_list']]

    mean_gesture = []
    for i in range(len(keyframes)):
        for j in range(keyframes[i]):
            mean_gesture.append(gestures[i][j])
    mean_gesture = np.array(mean_gesture)
    mean_gesture = np.mean(mean_gesture, axis=0)

    pad = np.array([mean_gesture])

    kf_gestures = []
    for i in range(len(keyframes)):
        kf = keyframes[i]
        gesture = gestures[i][:kf]
        while len(gesture) < max_keyframe:
            gesture = np.vstack([gesture, pad])       # add padding pose
        kf_gestures.append(gesture)
    kf_gestures = np.array(kf_gestures)

    dataset['X_train_gesture'] = kf_gestures

    with open(savedata_path, 'wb') as f:
        pickle.dump(dataset, f)


    print()

    # condirm mean pose
    new_pose = np.zeros([8, 3])
    new_pose[0] = mean_gesture[0]
    new_pose[2] = mean_gesture[1]
    new_pose[3] = new_pose[2] + mean_gesture[2]
    new_pose[4] = new_pose[3] + mean_gesture[3]
    new_pose[5] = mean_gesture[4]
    new_pose[6] = new_pose[5] + mean_gesture[5]
    new_pose[7] = new_pose[6] + mean_gesture[6]
    x = new_pose.T[0]
    y = new_pose.T[1]
    plt.figure()
    plt.plot([x[0], x[1]], [y[0], y[1]], linewidth=5)
    plt.plot([x[1], x[2]], [y[1], y[2]], linewidth=5)
    plt.plot([x[2], x[3]], [y[2], y[3]], linewidth=5)
    plt.plot([x[3], x[4]], [y[3], y[4]], linewidth=5)
    plt.plot([x[1], x[5]], [y[1], y[5]], linewidth=5)
    plt.plot([x[5], x[6]], [y[5], y[6]], linewidth=5)
    plt.plot([x[6], x[7]], [y[6], y[7]], linewidth=5)
    plt.show()

def add_timestamp():
    data_path = "./dataset/data_3-15keyframe_8joints_textlabel.pkl"
    lib_path = "./data/imagistic_gestures_clip.npy"
    save_data_path = "./dataset/data_3-15keyframe_8joints_textlabel_timestamp.pkl"

    data = pickle.load(open(data_path, "rb"))
    lib = np.load(lib_path, allow_pickle=True).item()

    print()


if __name__ == '__main__':
    # add_distance_matrix()
    # exit()

    # createTestData()
    # exit()

    # createNewData()
    # exit()

    add_timestamp()
    exit()

    fixPadding()
    exit()

    reannot_path = "./images/Gesture/gestures/data_5-12keyframe_1414gestures.xlsx"
    dataset_path = "./dataset/data_5-12keyframe_8joints_1414gestures.pkl"
    save_path = "./dataset/data_5-12keyframe_8joints_1000gestures_reannot_oldBERT.pkl"
    # embedding_file = "./data/glove.6B/glove.6B.300d.npy"
    bert_model = 'bert-base-uncased'
    

    df = pd.read_excel(reannot_path)
    dataset = pickle.load(open(dataset_path, "rb"))
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # embedding_index = np.load(embedding_file, allow_pickle=True).item()
    bert_model = BertModel.from_pretrained("bert-base-uncased")


    relative_gestures, x_text, keyframe_list = [], [], []
    gestures, valid_gestures, valid_gesture_ids, valid_texts, valid_remarks = [], [], [], [], []
    for i in range(len(df)):

        # not annotate yet
        if df.iloc[i]['reannotation'] == 0:
            continue

        # remove openpose mistaked data
        if df.iloc[i]['openpose'] == 'openpose mistake':
            continue

        if df.iloc[i]['gesture_type'] == 'beat' or df.iloc[i]['gesture_type'] == 'action' or df.iloc[i]['gesture_type'] == 'no-gesture':
            continue
        
        # long_text_id = tokenizer(df['texts'].iloc[i])['input_ids']
        # word_list = tokenizer.convert_ids_to_tokens(long_text_id)
        # remark_words = tokenizer.convert_ids_to_tokens(df['reannotation'].iloc[i])[1:-1]

        words = str(df['reannotation'].iloc[i])
        if words[-1] == ' ':
            words = words[:-1]
        word_list = nltk.word_tokenize(str.lower(words))

        relative_gestures.append(dataset['X_train_gesture'][i])
        keyframe_list.append(dataset['keyframe_list'][i])
        valid_gestures.append(dataset['valid_gestures'][i])
        gestures.append(dataset['gestures'][i])
        valid_gesture_ids.append(df['gesture_ids'].iloc[i])
        valid_texts.append(df['texts'].iloc[i])
        valid_remarks.append(df['reannotation'].iloc[i])

        # embed = tokenizer(str(df['reannotation'].iloc[i]))['input_ids']

        token = tokenizer(str(df['reannotation'].iloc[i]))['input_ids']
        # embed = bert_model(torch.tensor([token]))['pooler_output'][0].detach().numpy()
        hidden, _ = bert_model(torch.tensor([token]))
        embed = hidden[:,0,:][0].detach().numpy()

        x_text.append(embed)



    data = {}
    data['X_train_gesture'] = np.array(relative_gestures)
    data['X_test_gesture'] = np.array(relative_gestures)
    data['X_train_text'] = np.array(x_text)
    data['X_test_text'] = np.array(x_text)
    data['keyframe_list'] = np.array(keyframe_list)
    data['gestures'] = np.array(gestures)
    data['valid_gestures'] = np.array(valid_gestures)
    data['gesture_ids'] = np.array(valid_gesture_ids)
    data['texts'] = np.array(valid_texts)
    data['remarks'] = np.array(valid_remarks)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
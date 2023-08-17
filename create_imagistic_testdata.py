import os 
import json
import glob
import pickle
import librosa
import numpy as np
import pandas as pd
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

if __name__ == '__main__':
    kaist_data = "Z:/Human/b19-teshima/KAIST_dataset"
    annot_data = "./data/annotation_results_integrated_20220111.xlsx"
    data_dir = "Z:/Human/b19-teshima/TED_videos/VideoStorage/clip_poses"
    save_path = "./data/imagistic_gestures_clip.npy"

    df = pd.read_excel(annot_data)
    img_df = df[df['Gesture Type']=='Imagistic']

    k_data = pickle.load(open(kaist_data + "/ted_gesture_dataset_train.pickle", "rb"))
    vid_ids = np.array([v['vid'] for v in k_data])
    fps = 25

    gestures, labans, texts, remarks, g_ids, audios, words_timestamp = [], [], [], [], [], [], []
    c_ids = {}
    for i in tqdm(range(len(img_df))):
        g_id = img_df.iloc[i]['Gesture ID']
        clip_id = g_id[:g_id.rfind("_")]

        if clip_id in c_ids.keys():
            remarks[c_ids[clip_id]].append(img_df.iloc[i]['Remarks'])
            continue

        each_data_dir = data_dir + '/' + g_id[:11] + '/' + clip_id
        csv_path = each_data_dir + '/' + clip_id + '.csv'
        laban_path = each_data_dir + '/' + clip_id + '.json'
        audio_path = each_data_dir + '/' + clip_id + '.wav'

        if not (os.path.exists(csv_path) and os.path.exists(laban_path) and os.path.exists(audio_path)):
            print("{} is not found.".format(clip_id))
            continue

        audio, audio_sr = librosa.load(audio_path, mono=True, sr=16000, res_type='kaiser_fast')

        gesture = load_kinect_csv(csv_path)
        with open(laban_path, 'r') as f:
            laban = json.load(f)

        # text = img_df.iloc[i]['Text']
        remark = [img_df.iloc[i]['Remarks']]

        video_idx = np.where(vid_ids == g_id[:11])[0][0]
        clip_idx = int(clip_id[clip_id.rfind("_")+1:])
        words = k_data[video_idx]['clips'][clip_idx]['words']
        start = k_data[video_idx]['clips'][clip_idx]['start_frame_no']
        word_list = [[w[0], (w[1]-start)/fps, (w[2]-start)/fps] for w in words]

        text = " ".join([w[0] for w in words])

        gestures.append(gesture)
        labans.append(laban)
        texts.append(text)
        remarks.append(remark)
        g_ids.append(g_id)
        words_timestamp.append(word_list)
        c_ids[clip_id] = len(gestures)-1
        audios.append(audio)
    
    dic = {
        "gesture": gestures,
        "laban": labans,
        "text": texts,
        "remark": remarks,
        "audio": audios,
        "words_timestamp": words_timestamp,
        "clip_id": list(c_ids.keys())
    }

    np.save(save_path, dic)

    print("Done.")

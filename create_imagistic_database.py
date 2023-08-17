import os 
import json
import glob
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
    annot_data = "./data/annotation_results_integrated_20220111.xlsx"
    data_dir = "Z:/Human/b19-teshima/TED_videos/segmented_by_gesture"
    save_path = "./data/imagistic_gestures.npy"

    df = pd.read_excel(annot_data)
    img_df = df[df['Gesture Type']=='Imagistic']

    gestures, labans, texts, remarks, g_ids = [], [], [], [], []
    for i in tqdm(range(len(img_df))):
        g_id = img_df.iloc[i]['Gesture ID']
        each_data_dir = data_dir + '/' + g_id[:11] + '/' + g_id + '/' 
        csv_path = each_data_dir + '/' + g_id + '.csv'
        laban_path = each_data_dir + '/' + g_id + '.json'

        if not (os.path.exists(csv_path) and os.path.exists(laban_path)):
            print("{} is not found.".format(g_id))
            continue

        gesture = load_kinect_csv(csv_path)
        with open(laban_path, 'r') as f:
            laban = json.load(f)
        text = img_df.iloc[i]['Text']
        remark = img_df.iloc[i]['Remarks']

        gestures.append(gesture)
        labans.append(laban)
        texts.append(text)
        remarks.append(remark)
        g_ids.append(g_id)
    
    dic = {
        "gesture": gestures,
        "laban": labans,
        "text": texts,
        "remark": remarks,
        "gesture_id": g_ids
    }

    np.save(save_path, dic)

    print("Done.")

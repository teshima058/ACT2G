import os
import glob
import pickle
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from laban_distance import calcLabanDistance, loadKeyframeData, memoization

"""
Script to calculate distance matrix (for batch processing)

args:
  1. data_root_dir: video directory path
       {data_root_dir}
              |- {TED_Video_ID}
                      |- {TED_Video_ID_ClipID}
                              |- TED_Video_ID_ClipID.json:  labanotation key-frame file
                              |- TED_Video_ID_ClipID.mp4 :  original video

  2. save_path: path to save distance matrix file
"""

if __name__ == '__main__':

    dataset_path = "./dataset/data_5-12keyframe_8joints_1000gestures_reannot.pkl"
    data_root_dir = "Z:/Human/teshima/TED_videos/segmented_by_gesture/"
    diff_kf_penalty = 0.1
    save_path = "./data/distmat_1000reannot_pan={}.npy".format(diff_kf_penalty)

    dataset = pickle.load(open(dataset_path, "rb"))
    gesture_ids = dataset['gesture_ids']
    memo_data = memoization()

    keyframe_thresh = 10

    # Correct all labanotation json path
    print('Loading Labanotation Scores...')
    all_json_path = []
    keyframe_lengths = []
    for raw_video in glob.glob(data_root_dir + '**/'):
        video_id = os.path.basename(raw_video[:-1])
        for clip in glob.glob(raw_video + '**/'):
            clip_id = os.path.basename(clip[:-1])
            if not clip_id in gesture_ids:
                continue
            json_path = clip + clip_id + '.json'
            _, keyframe_list = loadKeyframeData(json_path)
            keyframe_lengths.append(len(keyframe_list))
            all_json_path.append(json_path)
    print('detected {} files'.format(len(all_json_path)))

    # # Just to confirm keyframe num 
    # df = pd.DataFrame(keyframe_lengths)
    # df.to_excel("./tmp.xlsx")

    # Calculate all combinations
    print('Calculating Distance Matrix...')
    vid_num = len(all_json_path)
    distance_mat = np.zeros([vid_num, vid_num])
    for i in tqdm(range(vid_num)):
        for j in range(vid_num):
            if i >= j:
                continue
            
            # Load json data
            _, keyframe_list1 = loadKeyframeData(all_json_path[i])
            _, keyframe_list2 = loadKeyframeData(all_json_path[j])

            # If the number of keyframes is very different, the distance is (5 joint_num) * (min_keyframe_num) + (difference of two keyframes)
            diff_keyframe_num = abs(len(keyframe_list1) - len(keyframe_list2))
            if diff_keyframe_num <= keyframe_thresh:
                distance_mat[i][j] = calcLabanDistance(all_json_path[i], all_json_path[j], memo_data=memo_data, diff_kf_penalty=diff_kf_penalty)
            else:
                min_keyframe_num = min(len(keyframe_list1), len(keyframe_list2))
                distance_mat[i][j] = 5 * min_keyframe_num + diff_keyframe_num

    for i in range(vid_num):
        for j in range(vid_num):
            if i < j:
                continue
            distance_mat[i][j] = distance_mat[j][i]
    
    # Save data
    data = {"json_list":all_json_path, "dist_mat":distance_mat}
    np.save(save_path, data)
    print('Output Data to', save_path)

    # For get the distance matrix again...
    # data = np.load('./data_output/distmat.npy', allow_pickle=True).item()

    print("Finish.")
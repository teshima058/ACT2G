import re
import os
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from plotPose import Plot
from text2speech import text2speech

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


def main():
    
    # ---------- Test ----------
    testdata_path = "./dataset/testdata_5-12keyframe.pkl"
    test_data = pickle.load(open(testdata_path, "rb"))

    
    # text_length = np.array([len(t.split()) for t in test_data['texts']])
    # keyframes = np.array([len(kf) for kf in test_data['keyframe_list']])
    # plt.figure()
    # plt.scatter(text_length, keyframes)
    # plt.show()



    # Save Test GT Data
    for i in tqdm(range(len(test_data['texts']))):
        kf = len(test_data["keyframe_list"][i])
        frame = len(test_data['gestures'][i])
        text = test_data['texts'][i]
        gt_kf = test_data['X_train_gesture'][i][:kf]
        gt = test_data['valid_gestures'][i][:frame]
        filename = "{}_kf={}_{}".format(str(i).zfill(4), kf, re.sub(r'[\\/:*?"<>|]+','', text)[:20].replace(" ", "_"))

        save_kf_path = "./images/test/test_data/gt_keyframe/{}_kf{}_{}.gif".format(str(i).zfill(4), kf, filename)
        save_gt_path = "./images/test/test_data/gt/{}_kf{}_{}.gif".format(str(i).zfill(4), kf, filename)
        save_gt_mp4_path = "./images/test/test_data/gt_mp4/{}_kf{}_{}.mp4".format(str(i).zfill(4), kf, filename)

        # if os.path.exists(save_gt_path):
        #     continue

        # plotUpperBody2D(gt_kf, save_kf_path, isRelative=True, fps=5, frames=np.arange(kf))
        # plotUpperBody2D(gt, save_gt_path, isRelative=True, fps=25, frames=np.arange(len(gt)))
        plotUpperBody2D(gt, "./.tmp/tmp.mp4", isRelative=True, fps=25, frames=np.arange(len(gt)))

        # Text2Audio
        tmp_wav_path = "./.tmp/tmp.wav"
        text2speech(text, tmp_wav_path)
        if not os.path.exists(tmp_wav_path):
            return -1
        input_audio = tmp_wav_path

        cmd = ["ffmpeg", "-i", "./.tmp/tmp.mp4", "-i", tmp_wav_path, "-c:v", "copy", "-c:a", "aac", "-strict", \
            "experimental", "-map", "0:v", "-map", "1:a", save_gt_mp4_path , "-y", "-hide_banner", "-loglevel", "error"]
        subprocess.call(cmd)

    exit()


            
if __name__ == '__main__':
    main()

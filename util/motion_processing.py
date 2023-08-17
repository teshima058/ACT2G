import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import signal, interpolate

from .adjust_motion2audio import interpolateKeypose


def interpolateKeypose(keyposes, frame=60, s=None):   # [keyframe, joint, xyz]
    keyposes = keyposes.transpose([1, 2, 0])
    new_poses = []
    for pose in keyposes:
        tck, u = interpolate.splprep(pose, k=3, s=s)
        new = interpolate.splev(np.linspace(0,1,frame), tck, der=0)
        new_poses.append(new)
    new_poses = np.array(new_poses).transpose([2, 0, 1])
    return new_poses


def smoothingPose(pose3d, kernel_size=5):
    n_joints = pose3d.shape[1]
    data = pose3d.reshape(-1, n_joints*3).T
    new_pose = []
    for i in range(len(data)):
        filtered = gaussian_filter1d(data[i], kernel_size)
        new_pose.append(filtered)
    new_pose = np.array(new_pose).T.reshape(-1, n_joints, 3)
    return new_pose


def motionAdjustment(motion, adjust_frame, types=None):
    if len(motion) == adjust_frame:
        adjusted_motion = motion
    
    # Interpolation
    elif len(motion) < adjust_frame:
        adjusted_motion = np.zeros([adjust_frame, motion.shape[1], 3])
        adjusted_motion[0] = motion[0]
        if types:
            adjusted_types = [""] * adjust_frame
            adjusted_types[0] = types[0]
            adjusted_types[len(adjusted_types)-1] = types[len(types)-1]
        adjusted_motion[len(adjusted_motion)-1] = motion[len(motion)-1]
        interval = (adjust_frame - 2) / (len(motion) - 2)
        interval_sum, count = 0, 0
        e_index, tmp = [], []
        isEmpty = False
        for i in range(1, len(adjusted_motion)-1):
            if i > interval_sum:
                adjusted_motion[i] = motion[count]
                if types:
                    if count < len(types):  # To Do
                        adjusted_types[i] = types[count]
                interval_sum += interval
                count += 1
            else:
                e_index.append(i)

        empty_index, tmp = [], []
        for i in range(len(e_index)):
            if i == len(e_index) - 1:
                tmp.append(e_index[i])
                empty_index.append(tmp)
            elif e_index[i]+1 == e_index[i+1]:
                tmp.append(e_index[i])
            else:
                tmp.append(e_index[i])
                empty_index.append(tmp)
                tmp = []

        for e in empty_index:
            unit = (adjusted_motion[e[len(e)-1]+1] - adjusted_motion[e[0]-1]) / (len(e) + 1)
            cnt = 1
            for i in range(e[0], e[len(e)-1] + 1):
                adjusted_motion[i] = adjusted_motion[e[0]-1] + unit*cnt
                if types:
                    adjusted_types[i] = adjusted_types[e[0]-1]
                cnt += 1

    # Sampling
    else:
        adjusted_motion = []
        adjusted_types = []
        decrease_frame = len(motion) - adjust_frame
        sampling_frames = [int(n) for n in np.arange(0, len(motion), len(motion) / decrease_frame)]
        sampling_frames = sampling_frames[:decrease_frame]

        if len(sampling_frames) != decrease_frame:
            print('Error: samping failed', file=sys.stderr)
            sys.exit()
            
        for i in range(len(motion)):
            if i in sampling_frames:
                continue
            adjusted_motion.append(motion[i])
            if types:
                adjusted_types.append(types[i])
    
    if types:
        return np.array(adjusted_motion), adjusted_types
    else:
        return np.array(adjusted_motion)


# motions.shape = ({motion_num}, {frame_num}, {joint_num}, {xyz})
def linearInterpolation(motions, interpolate_frame=20, mean_pose=None):
    def interpolate(m1, m2, interpolate_frame):
        first_pose = m1[len(m1)-1]
        last_pose = m2[0]

        pose = []
        for i in range(len(first_pose)):
            joint = []
            for j in range(len(first_pose[i])):
                diff = (last_pose[i][j] - first_pose[i][j]) / (interpolate_frame + 1)
                frame = []
                pos = first_pose[i][j]
                for k in range(interpolate_frame):
                    pos += diff
                    frame.append(pos)
                joint.append(frame)
            pose.append(joint)
        pose = np.array(pose)
        pose = pose.transpose(2, 0, 1)
        
        return pose
    
    interpolated_motions = []
    for i in range(len(motions)):
        if i == 0:
            for m in motions[i]:
                interpolated_motions.append(m)
            continue

        interpo_motion = interpolate(motions[i-1], motions[i], interpolate_frame)

        for m in interpo_motion:
            interpolated_motions.append(m)

        for m in motions[i]:
            interpolated_motions.append(m)

    interpolated_motions = np.array(interpolated_motions)
    return interpolated_motions


def CMUPose2KinectData(pose3d, save_csv=None, fps=25, isConvert=True):

    cmu2kinect = [2, 20, 8, 9, 10, 4, 5, 6]

    if isConvert:
        kinect_data = np.zeros((len(pose3d), 25, 3))
        for i,pose in enumerate(pose3d):
            # Replacing joints
            for j in range(len(cmu2kinect)):
                kinect_data[i][cmu2kinect[j]] = pose[j]
            # spinMid -> Midpoint between midhip and neck
            kinect_data[i][1] = (kinect_data[i][0] + kinect_data[i][20]) / 2
            # head -> same as nose
            kinect_data[i][3] = kinect_data[i][2]
            # toes of both feet -> both ankles
            kinect_data[i][15] = kinect_data[i][14]
            kinect_data[i][19] = kinect_data[i][18]

            # Rotate x and y axis
            for j in range(len(kinect_data[i])):
                kinect_data[i][j][0] *= -1
                kinect_data[i][j][1] *= -1
            
            # Recoordinate SpineMid to (0, 0, 0)
            spineMid = kinect_data[i][1].copy()
            for j in range(len(kinect_data[i])):
                kinect_data[i][j] -= spineMid

            # Normalized so that the distance between midspine and midhip is 0.25
            spine_hip_length = np.linalg.norm(kinect_data[i][1] - kinect_data[i][0])
            kinect_data[i] = kinect_data[i] * 0.25 / spine_hip_length
    else:
        kinect_data = pose3d

    # Save CSV
    if save_csv:
        # Timestamp for kinect data
        msec = 1/fps * 1000
        timestamp = np.arange(0, len(pose3d)*msec, msec)
        timestamp = timestamp[:len(pose3d)]
        timestamp = timestamp.reshape(-1, 1)
        
        # Extra data for kinect data
        # 0 -> no-detection,    2 -> detected
        kinect_extra = np.zeros((len(pose3d), 25))
        for i in range(len(kinect_extra)):
            for j in range(len(kinect_extra[i])):
                if j in cmu2kinect or j in [1, 3, 15, 19]:
                    kinect_extra[i][j] = 2
                else:
                    kinect_extra[i][j] = 0

        # Concat timestamp and data
        kinect_extra = kinect_extra.reshape(-1, 25, 1)
        tmp = np.concatenate([kinect_data, kinect_extra], axis=2)
        tmp = tmp.reshape(-1, 25 * 4)
        kinect_csv_data = np.concatenate([timestamp, tmp], axis=1)


        # Save
        kinect_csv_data = pd.DataFrame(kinect_csv_data)
        for i in range(0, 101, 4):
            kinect_csv_data[i] = kinect_csv_data[i].astype('int')
        kinect_csv_data.to_csv(save_csv, header=False, index=False)
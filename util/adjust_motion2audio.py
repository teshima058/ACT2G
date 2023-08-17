import os
import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d
import parselmouth as pm


def interpolateKeypose(keyposes, frame=60):   # [keyframe, joint, xyz]
    keyposes = keyposes.transpose([1, 2, 0])
    new_poses = []
    for pose in keyposes:
        tck, u = interpolate.splprep(pose, k=3)
        new = interpolate.splev(np.linspace(0,1,frame), tck, der=0)
        new_poses.append(new)
    new_poses = np.array(new_poses).transpose([2, 0, 1])
    return new_poses


def compute_prosody(audio_filename, time_step=0.04, kernel_size=1):
    audio = pm.Sound(audio_filename)

    # Extract pitch and intensity
    pitch = audio.to_pitch(time_step=time_step)
    intensity = audio.to_intensity(time_step=time_step)

    # Evenly spaced time steps
    times = np.arange(0, audio.get_total_duration() - time_step, time_step)

    # Compute prosodic features at each time step
    pitch_values = np.nan_to_num(
        np.asarray([pitch.get_value_at_time(t) for t in times]))
    intensity_values = np.nan_to_num(
        np.asarray([intensity.get_value(t) for t in times]))

    intensity_values = np.clip(
        intensity_values, np.finfo(intensity_values.dtype).eps, None)

    # Normalize features [Chiu '11]
    pitch_norm = np.clip(np.log(pitch_values + 1) - 4, 0, None)
    intensity_norm = np.clip(np.log(intensity_values) - 3, 0, None)

    fileterd_pitch = gaussian_filter1d(pitch_norm, kernel_size)
    fileterd_intensity = gaussian_filter1d(intensity_norm, kernel_size)

    return fileterd_pitch, fileterd_intensity, times

def adjust_audio(keyposes, audio_path):
    audio_gaussian = 1
    audio_feature='intensity'

    # Extract audio feature
    pitch, intensity, time = compute_prosody(audio_path)

    if audio_feature == 'pitch':
        audio_wave = pitch
    elif audio_feature == 'intensity':
        audio_wave = intensity

    # Cut the last part that doesn't say anything.
    for i in range(len(audio_wave)-1, 0, -1):
        if audio_wave[i] > 1e-4:
            end_idx = i
            break
    audio_wave = audio_wave[:end_idx]
    audio_wave = gaussian_filter1d(audio_wave, audio_gaussian)

    audio_keyframe = signal.argrelmax(audio_wave, order=1)[0]
    audio_keyframe_num = max(len(audio_keyframe), 1)
    audio_length = len(audio_wave)

    # # spiline interpolation
    # pose3d = interpolateKeypose(keyposes, frame=audio_length)

    # liunear interpolation
    motion = keyposes.reshape([keyposes.shape[0], 1, keyposes.shape[1], keyposes.shape[2]]).detach().cpu().numpy()
    keyposes = keyposes.detach().cpu().numpy()
    pose3d = linearInterpolation(motion, int(audio_length / (motion.shape[0] - 1)))

    keyframe = np.array([np.argmin(np.sum(np.sum(np.abs(pose3d - keyposes[i]), axis=1), axis=1)) for i in range(len(keyposes))])

    valid_audio_keyframe = []
    for kf in keyframe:
        idx = np.argmin(np.abs(audio_keyframe - kf))
        valid_audio_keyframe.append(audio_keyframe[idx])


    # # Adjust motion to audio
    # pre_mkf, pre_akf = 0, 0
    # for mkf, akf in zip(keyframe, valid_audio_keyframe):
    #     length = akf - pre_akf
    #     motion = motionAdjustment(pose3d[pre_mkf:mkf], length)
    #     # print("{} -> {}".format(len(pose3d[pre_mkf:mkf]), len(motion)))
    #     for m in motion:
    #         adjusted_motion.append(m)
    #     pre_mkf = mkf
    #     pre_akf = akf
    # motion = motionAdjustment(pose3d[pre_mkf:], len(audio_wave)-pre_akf)

    # # # Confirm
    # new_energy = totalEnergy(adjusted_motion)
    # new_keyframe = list(signal.argrelmax(new_energy, order=1)[0])
    # now = datetime.datetime.now()
    # save_path = "./.tmp/beat_energy_{}.png".format(now.strftime('%Y%m%d_%H%M%S'))
    # plotTransition([new_energy, audio_wave], ['motion energy', 'audio'], [new_keyframe, audio_keyframe], save_path=save_path)
    
    return pose3d
        

def motionAdjustment(motion, adjust_frame, joint_num=7, types=None):
    if len(motion) == adjust_frame:
        adjusted_motion = motion
    
    # Interpolation
    elif len(motion) < adjust_frame:
        adjusted_motion = np.zeros([adjust_frame, joint_num, 3])
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
        if decrease_frame == 1:
            decrease_interval = len(motion) / 2 + 1
        else:
            decrease_interval = len(motion) / decrease_frame
        for i in range(len(motion)):
            if i != 0 and i % decrease_interval == 0:
                continue
            adjusted_motion.append(motion[i])
            if types:
                adjusted_types.append(types[i])
            if len(adjusted_motion) == adjust_frame:
                break
    
    if types:
        return np.array(adjusted_motion), adjusted_types
    else:
        return np.array(adjusted_motion)


def plotTransition(lists, labels, keyframes=None, sr=None, title=None, save_path=None):
    figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (8, 4)
    for i in range(len(lists)):
        if sr is None:
            if keyframes:
                plt.plot(lists[i], label=labels[i], marker="o", markevery=keyframes[i])
            else:
                plt.plot(lists[i], label=labels[i])
        else:
            time = np.linspace(0, len(lists[i])/sr, len(lists[i]))
            if keyframes:
                plt.plot(time, lists[i], label=labels[i], marker="o", markevery=keyframes[i])
            else:
                plt.plot(time, lists[i], label=labels[i])

    plt.legend()
    plt.xlabel('Time')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    plt.rcParams["figure.figsize"] = figsize


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


if __name__ == '__main__':
    # wav_path = "./data/_ryJK294Psw_14/_ryJK294Psw_14.wav"
    # wav_path = "./data/1oNlTrLIjU4_8/1oNlTrLIjU4_8.wav"
    # wav_path = "./data/_vBggxCNNno_4/_vBggxCNNno_4.wav"
    wav_path = "./data/38OUCtzkT4Q_7_0.wav"

    beat_library_path = "./data/beat_library_20211006.npy"
    audio_feature = 'intensity'
    audio_gaussian = 1

    csv_save_path = "./output/csv/{}.csv".format(os.path.basename(wav_path)[:-4])
    mp4_save_path = "./output/mp4/{}.mp4".format(os.path.basename(wav_path)[:-4])
    
    # Generate



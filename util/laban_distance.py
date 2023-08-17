import json
import math
import numpy as np
import argparse
from itertools import combinations

"""
Script to calculate the distance between two labanotations

args:
  1. Json file with labanotation keyframes
  2. The other json file you want to calculate the distance
"""

# Euler angles
angle = {'High':{}, 'Normal':{}, 'Low':{}}
angle['High']['Forward']            = [   90,  45]
angle['High']['Right Forward']      = [   45,  45]
angle['High']['Right']              = [    0,  45]
angle['High']['Right Backward']     = [  -45,  45]
angle['High']['Backward']           = [  -90,  45]
angle['High']['Left Backward']      = [ -135,  45]
angle['High']['Left']               = [  180,  45]
angle['High']['Left Forward']       = [  135,  45]
angle['High']['Place']              = [   90,  90]
angle['Normal']['Forward']          = [   90,   0]
angle['Normal']['Right Forward']    = [   45,   0]
angle['Normal']['Right']            = [    0,   0]
angle['Normal']['Right Backward']   = [  -45,   0]
angle['Normal']['Backward']         = [  -90,   0]
angle['Normal']['Left Backward']    = [ -135,   0]
angle['Normal']['Left']             = [  180,   0]
angle['Normal']['Left Forward']     = [  135,   0]
angle['Low']['Forward']             = [   90, -45]
angle['Low']['Right Forward']       = [   45, -45]
angle['Low']['Right']               = [    0, -45]
angle['Low']['Right Backward']      = [  -45, -45]
angle['Low']['Backward']            = [  -90, -45]
angle['Low']['Left Backward']       = [ -135, -45]
angle['Low']['Left']                = [  180, -45]
angle['Low']['Left Forward']        = [  135, -45]
angle['Low']['Place']               = [   90, -90]

# direction indexes
directions = {'High':{}, 'Normal':{}, 'Low':{}}
directions['High']['Forward']            = 0
directions['High']['Right Forward']      = 1
directions['High']['Right']              = 2
directions['High']['Right Backward']     = 3
directions['High']['Backward']           = 4
directions['High']['Left Backward']      = 5
directions['High']['Left']               = 6
directions['High']['Left Forward']       = 7
directions['High']['Place']              = 8
directions['Normal']['Forward']          = 9
directions['Normal']['Right Forward']    = 10
directions['Normal']['Right']            = 11
directions['Normal']['Right Backward']   = 12
directions['Normal']['Backward']         = 13
directions['Normal']['Left Backward']    = 14
directions['Normal']['Left']             = 15
directions['Normal']['Left Forward']     = 16
directions['Low']['Forward']             = 17
directions['Low']['Right Forward']       = 18
directions['Low']['Right']               = 19
directions['Low']['Right Backward']      = 20
directions['Low']['Backward']            = 21
directions['Low']['Left Backward']       = 22
directions['Low']['Left']                = 23
directions['Low']['Left Forward']        = 24
directions['Low']['Place']               = 25


def vector(angle):
    x = math.cos(math.radians(angle[1])) * math.cos(math.radians(angle[0]))
    y = math.sin(math.radians(angle[1]))
    z = math.cos(math.radians(angle[1])) * math.sin(math.radians(angle[0]))
    return np.array([x,y,z])


def calcJointDistance(angle1, angle2):
    # calculate cos similarity
    v1 = vector(angle1)
    v2 = vector(angle2)
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # normalize (same direction -> 0, opposite direction -> 1)
    dist = abs(cos_sim * -1 + 1) / 2
    return dist


def memoization():
    idx2dir = []
    for v in list(angle.keys()):
        for h in list(angle[v].keys()):
            idx2dir.append(v + '|' + h)
    n = len(idx2dir) # 26

    memo = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            score1 = idx2dir[i].split('|')
            score2 = idx2dir[j].split('|')
            memo[i][j] = calcJointDistance(angle[score1[0]][score1[1]], angle[score2[0]][score2[1]])
    
    return memo


def loadKeyframeData(json_file):
    with open(json_file, 'r') as f:
        laban_score = json.load(f)

    gesture_name = list(laban_score.keys())[0]
    laban_score = laban_score[gesture_name]
    keyframe_list = list(laban_score.keys())
    
    return laban_score, keyframe_list


def calcLabanDistance(json_file1, json_file2, diff_kf_penalty=0.1, memo_data=None):

    def JointDistance(joint, score1, score2, memo=None):
        if memo is not None:
            idx1 = directions[score1[joint][1]][score1[joint][0]]
            idx2 = directions[score2[joint][1]][score2[joint][0]]
            return memo[idx1][idx2]
        else:
            angle1 = angle[score1[joint][1]][score1[joint][0]]
            angle2 = angle[score2[joint][1]][score2[joint][0]]
            return calcJointDistance(angle1, angle2)

    # Load json data
    laban_score1, keyframe_list1 = loadKeyframeData(json_file1)
    laban_score2, keyframe_list2 = loadKeyframeData(json_file2)

    # Swap so that 1 is more than 2
    if len(keyframe_list1) < len(keyframe_list2):
        json_file1, json_file2 = json_file2, json_file1
        laban_score1, laban_score2 = laban_score2, laban_score1
        keyframe_list1, keyframe_list2 = keyframe_list2, keyframe_list1

    # Iterate nCr times ( [Number of keyframes with more keyframes]C[Difference in the number of keyframes] ) 
    diff_keyframe = len(keyframe_list1) - len(keyframe_list2)
    calc_iter = list(combinations(keyframe_list1, diff_keyframe))

    # Weighting keyframe 
    # Keyframes with large motion are given larger weights in the distance calculation.
    # lw_weight, rw_weight = [], []
    # lw_tmp, rw_tmp = [], []
    # for i in range(1, len(keyframe_list2)):
    #     bef_score = laban_score2[keyframe_list2[i-1]]
    #     aft_score = laban_score2[keyframe_list2[i]]
    #     lw = JointDistance('left wrist', bef_score, aft_score, memo_data) / 2 + 0.5
    #     rw = JointDistance('right wrist', bef_score, aft_score, memo_data) / 2 + 0.5

    #     if i == 1:
    #         lw_weight.append(lw)
    #         rw_weight.append(rw)
    #         lw_tmp.append(lw)
    #         rw_tmp.append(rw)
    #         if i == len(keyframe_list2) - 1:
    #             lw_weight.append(lw)
    #             rw_weight.append(rw)
    #     elif i == len(keyframe_list2) - 1:
    #         lw_weight.append(max(lw, lw_tmp[i-2]))
    #         rw_weight.append(max(rw, rw_tmp[i-2]))
    #         lw_weight.append(lw_weight[len(lw_weight)-1])
    #         rw_weight.append(rw_weight[len(rw_weight)-1])
    #     else:
    #         lw_weight.append(max(lw, lw_tmp[i-2]))
    #         rw_weight.append(max(rw, rw_tmp[i-2]))
    #         lw_tmp.append(lw)
    #         rw_tmp.append(rw)

    distance = float('inf')
    for missing_frame in calc_iter:

        new_keyframe_list = keyframe_list1.copy()   
        for kf in missing_frame:
            new_keyframe_list.remove(kf)  # Remove uncorresponded keyframes

        head_sum, left_elbow_sum, left_wrist_sum, right_elbow_sum, right_wrist_sum = 0, 0, 0, 0, 0

        # Calculate the distance for each keyframe
        for i in range(len(new_keyframe_list)):
            score1 = laban_score1[new_keyframe_list[i]]
            score2 = laban_score2[keyframe_list2[i]]

            # Calculate the distance between head joints
            # head_sum += JointDistance('head', score1, score2, memo_data)

            # Calculate the distance between left elbow joints
            # left_elbow_sum += JointDistance('left elbow', score1, score2, memo_data)

            # Calculate the distance between left wrist joints
            left_wrist_sum += JointDistance('left wrist', score1, score2, memo_data)
            # left_wrist_sum += JointDistance('left wrist', score1, score2, memo_data) * lw_weight[i]

            # Calculate the distance between right elbow joints
            # right_elbow_sum += JointDistance('right elbow', score1, score2, memo_data)

            # Calculate the distance between right wrist joints
            right_wrist_sum += JointDistance('right wrist', score1, score2, memo_data)
            # right_wrist_sum += JointDistance('right wrist', score1, score2, memo_data) * rw_weight[i]


        # Total distance of each joint
        joints_sum = head_sum + left_elbow_sum + left_wrist_sum + right_elbow_sum + right_wrist_sum

        # Minimum distance
        if joints_sum < distance:
            distance = joints_sum
            # print(missing_frame)

    # Add penalty for missing keyframes
    distance += diff_keyframe * diff_kf_penalty

    return distance


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate the distance between two labanotations')

    parser.add_argument('json_file1', help='Json file with labanotation keyframes')
    parser.add_argument('json_file2', help='The other json file you want to measure the distance')
    args = parser.parse_args()

    json_file1 = args.json_file1
    json_file2 = args.json_file2

    # memo data for DP
    memo_data = memoization()
    
    distance = calcLabanDistance(json_file1, json_file2, memo_data=memo_data)

    print("Distance : {}".format(distance))
import math
import torch
import socket
import argparse
import pickle
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import BertConfig
import pickle


class Gesture(object):
    def __init__(self, train, gesture, text, text_label, kf_list, opt):
        self.gesture = gesture
        self.text = torch.tensor(text)
        self.text_label = torch.tensor(text_label)
        self.kf_list = torch.tensor(kf_list)


    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.gesture.shape[0]

    def __getitem__(self, index):
        gesture = self.gesture[index] 
        text = self.text[index] 
        text_label = self.text_label[index] 
        kf_list = self.kf_list[index] 
        return {"gesture": gesture, "text": text, "text_label": text_label, 'kf_list': kf_list, "index": index}


class Gesture_GVAE(object):
    def __init__(self, train, gesture, kf_list, opt):
        self.gesture = gesture
        self.kf_list = torch.tensor(kf_list)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.gesture.shape[0]

    def __getitem__(self, index):
        gesture = self.gesture[index] 
        kf_list = self.kf_list[index] 
        return {"gesture": gesture, 'kf_list': kf_list, "index": index}


def load_dataset(opt):
    data = pickle.load(open(opt.dataset, "rb"))

    if opt.frames > 30:
        X_train_gesture, X_test_gesture = data['align_gestures'], data['align_gestures']
        kf_list = np.array([len(d) for d in data['gestures']])
    else:
        X_train_gesture, X_test_gesture = data['X_train_gesture'], data['X_test_gesture']
        kf_list = np.array([len(k) for k in data['keyframe_list']])
    X_train_text, X_test_text = data['X_train_text'], data['X_test_text']
    positive = data['positive']
    text_label = data['text_labels']

    train_data = Gesture(train=True, gesture=X_train_gesture, text=X_train_text, text_label=text_label, kf_list=kf_list, opt=opt)
    test_data = Gesture(train=False, gesture=X_test_gesture, text=X_test_text, text_label=text_label, kf_list=kf_list, opt=opt)
    print("finish loading!")

    return train_data, test_data, positive


def load_dataset_GVAE(opt):
    data = pickle.load(open(opt.dataset, "rb"))

    if opt.frames > 200:
        X_train_gesture, X_test_gesture = data['valid_gestures'], data['valid_gestures']
        kf_list = np.array([len(k) for k in data['gestures']])
    else:
        X_train_gesture, X_test_gesture = data['X_train_gesture'], data['X_train_gesture']
        kf_list = np.array([len(k) for k in data['keyframe_list']])

    train_data = Gesture_GVAE(train=True, gesture=X_train_gesture, kf_list=kf_list, opt=opt)
    test_data = Gesture_GVAE(train=False, gesture=X_test_gesture, kf_list=kf_list, opt=opt)
    print("finish loading!")

    return train_data, test_data


def clear_progressbar():
    print("\033[2A")
    print("\033[2K")
    print("\033[2A")


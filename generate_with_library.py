import os
import re
import glob
import shutil
import subprocess
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.neighbors import NearestNeighbors

from model import ACT2G
# from model_wokf import ACT2G
from plotPose import Plot
from util.text2speech import text2speech
from util.adjust_motion2audio import adjust_audio, compute_prosody
from util.motion_processing import smoothingPose, motionAdjustment, linearInterpolation, CMUPose2KinectData

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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
        p = Plot((-0.5, 0.5), (-0.4, 0.6))
    else:
        pose2d = np.delete(pose3d, 2, 2).reshape([-1, pose3d.shape[1]*2])
        p = Plot((-0.5, 0.5), (-0.6, 0.4))

    if frames is not None:
        pose2d = np.append(pose2d.T, [frames], axis=0).T

    anim = p.animate(pose2d, 1000/fps)
    p.save(anim, save_path, fps=fps)
    plt.close()


def plotResult(X, labels, text=None):
    unique_labels = set(labels)
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [cmap[i%len(cmap)] for i in range(len(labels))]
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(X)
    for k, col in zip(unique_labels, colors):
        class_member_mask = labels == k
        xy = X_reduced[class_member_mask]
        if text is not None:
            texts = np.array(text)[class_member_mask]
            for i in range(0, len(xy), 1):
                plt.text(xy[i][0], xy[i][1], texts[i], color=col)
            plt.xlim(-40, 70)
            plt.ylim(-45, 45)
        else:
            plt.scatter(xy[:, 0], xy[:, 1], marker="${}$".format(k), label=k, color=col)
    plt.title("KMeans Clustering")
    plt.show()
    plt.close()


def visualizePrediction(tokens, pred, labels=None, save_path=None):
    fig, ax = plt.subplots()
    idx = tokens.index('[PAD]')
    # idx = len(tokens)
    if labels is not None:
        data = np.array([pred[:idx], labels[:idx]]).T
        ax.set_xticklabels(['Predicted', 'GT'], rotation=0)
    else:
        data = np.array([pred[:idx]]).T
        ax.set_xticklabels(['Predicted'], rotation=0)
    sns.heatmap(data=data, cmap='OrRd', annot=True, vmin=0, vmax=max(data)[0])
    ax.set_yticks(np.arange(idx)+0.5)
    ax.set_yticklabels(tokens[:idx], rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def generate(opt, text, lib, model, focus_word=None, input_audio=None, filename=None, tokenizer=None,
save_csv_dir=None, save_mp4_dir=None, save_attention_dir=None, text_label=None, TEXT_SPLIT=8):
    
    INTERPOLATE_FRAME = 7
    SMOOTHING_WINDOW = 1
    FPS = 25

    tmp_wav_path = "./.tmp/audio/" + filename + '.wav'
    os.makedirs(os.path.dirname(tmp_wav_path), exist_ok=True)

    # Text2Audio
    if not input_audio:
        text2speech(text, tmp_wav_path)
        if not os.path.exists(tmp_wav_path):
            return -1
        input_audio = tmp_wav_path

    pitch, intensity, time = compute_prosody(input_audio)
    # Cut the last part that doesn't say anything.
    for i in range(len(intensity)-1, 0, -1):
        if intensity[i] > 1e-4:
            end_idx = i
            break
    frame_num = end_idx
    
    if tokenizer is None:
        # model setting
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if focus_word:
        focus_tokens = tokenizer(focus_word)['input_ids'][1:-1]

    count = 0
    each_texts = []
    while(count<len(text.split())):
        each_texts.append(" ".join(text.split()[count:count+TEXT_SPLIT]))
        count += TEXT_SPLIT
    
    gesture_seqs, nn_texts, nn_gesture_ids, x_texts, attns = [], [], [], [], []
    for t in each_texts:
    
        x_text = torch.tensor(tokenizer(t)['input_ids'])

        attention = torch.zeros(x_text.shape[0])
        if focus_word:
            wheres = []
            for t in focus_tokens:
                for w in torch.where(x_text == t)[0]:
                    wheres.append(w)
            for i in range(len(x_text)):
                if i in wheres[1:]:
                # if i in wheres:
                    attention[i] = 0.5
                else:
                    attention[i] = 0.1

            attention = torch.cat((attention, torch.zeros(opt.text_length - attention.shape[0])))
            attention = attention.unsqueeze(0)
            attention = attention.unsqueeze(2).cuda()
        else:
            attention = None
            
        x_text = torch.cat((x_text, torch.zeros(opt.text_length - x_text.shape[0])))
        x_text = x_text.unsqueeze(0).cuda()
        x_texts.append(x_text)


        # inference
        recon_g, text_feature, attn, kf_pred = model.generate_gesture(x_text)
        # recon_g, text_feature, attn = model.generate_gesture(x_text, attention)
        text_feature = text_feature.detach().cpu().numpy()[0]
        
        knn_model = NearestNeighbors(n_neighbors=1).fit(lib['g_features']) 
        # select cluster by nearest neighbor
        dists, index = knn_model.kneighbors(text_feature.reshape(1, -1))
        index = index[0][0]

        nn_texts.append(lib['texts'][index])
        nn_gesture_ids.append(lib['gesture_ids'][index])
        attns.append(attn)

        index_list = np.where(lib['g_labels'] == lib['g_labels'][index])[0]
        gesture_list = lib['gestures'][index_list]
        speed_list = lib["g_speeds"][index_list]
        length_list = np.array([len(g) for g in gesture_list])
        sp_len_list = length_list * speed_list

        # # Select most closest time gesture
        # frame_list = np.array([len(g) for g in gesture_list])
        # idx = np.argmin(np.abs(frame_list - frame_num)) 

        # Select randomly
        # idx = random.randrange(len(gesture_list))

        # Select randomly among the top k speeds.
        k = 5
        # candidates = np.argpartition(-speed_list, k)[:k]
        candidates = np.argpartition(-sp_len_list, k)[:k]
        idx = candidates[random.randrange(k)]

        gesture_seqs.append(gesture_list[idx])

    # Adjust to match fps
    gesture = linearInterpolation(gesture_seqs, interpolate_frame=INTERPOLATE_FRAME)
    gesture = motionAdjustment(gesture, frame_num)
    gesture = smoothingPose(gesture, kernel_size=SMOOTHING_WINDOW)


    ############   Save Gestures   ############
    if save_mp4_dir:
        os.makedirs(save_mp4_dir, exist_ok=True)
        # fps = 25 if opt.frames > 30 else 3
        frames = np.arange(len(gesture))
        save_gesture_path = save_mp4_dir + '/' + filename + ".mp4"
        tmp_mp4_path = "./.tmp/mp4/" + filename + '.mp4'
        plotUpperBody2D(gesture, tmp_mp4_path, isRelative=False, fps=FPS, frames=frames)

        cmd = ["ffmpeg", "-i", tmp_mp4_path, "-i", input_audio, "-c:v", "copy", "-c:a", "aac", "-strict", \
            "experimental", "-map", "0:v", "-map", "1:a", save_gesture_path, "-y", "-hide_banner", "-loglevel", "quiet"]
        subprocess.call(cmd)

        print("Saved to {}".format(save_gesture_path))
    
    if save_csv_dir:
        save_csv_path = save_csv_dir + '/' + filename + ".csv"
        CMUPose2KinectData(gesture, save_csv=save_csv_path, fps=FPS, isConvert=False)


    ############   Visualize Attention Weight   ############
    if save_attention_dir:
        os.makedirs(save_attention_dir, exist_ok=True)
        for i in range(len(x_texts)):
            a = attns[i].detach().cpu().numpy()[0]
            tokens = tokenizer.convert_ids_to_tokens(x_texts[i][0])
            save_attention_path = save_attention_dir + '/' + filename + "_{}.png".format(str(i).zfill(3))
            visualizePrediction(tokens, a, labels=text_label, save_path=save_attention_path)

        save_text_path = save_attention_dir + '/' + filename + ".txt"
        with open(save_text_path, 'w') as f:
            f.write("Input Text: {}\n\n".format(text))
            for i in range(len(each_texts)):
                f.write("Splited Text: {}\n".format(each_texts[i]))
                f.write("NN Text: {}\n".format(nn_texts[i]))
                f.write("NN GestureID: {}\n\n".format(nn_gesture_ids[i]))
    
    return 0



def main():
    
    # input_text = "business makes more money if they don't have a safe working environment. "
    # input_text = "business makes more money if they dont have a safe workoutputing environment . thats been the conventional wisdom ."
    # input_text = "thats been the conventional wisdom . if they dont have a safe working environment, business makes more money ."
    # input_text = "company makes more money if I didnt have a safe working environment. I have the conventional wisdom."
    # input_text = "company creates more money if we have a safe working space."
    # input_text = "company destroys more money if we have a safe resting space. thats been the knowledge."
    # input_text = "I placed those two long sticks at the top."
    # input_text = "I open the door and then close it"
    input_text = "Today is six degrees higher than yesterday."

    focus_word = "there"
    # focus_word = None


    lib_path = "./data/library_ACT2G_α=10.0_β=2.0_γ=1.0_margin=20.0_re.npy"
    # lib_path = "./data/library_ACT2G_woattn_α=10.0_β=2.0_γ=1.0_margin=20.0.npy"

    lib = np.load(lib_path, allow_pickle=True).item()
    opt = lib['model']['option']
    opt.batch_size = 1
    model = ACT2G(opt).cuda()
    model.load_state_dict(lib['model']['model'])
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



    # # ----------- Inference ----------
    # save_dir = "./images/reconst/{}/generate/gesture_libary/".format(os.path.basename(os.path.dirname(opt.model_path)))
    # save_mp4_dir = save_dir + "gesture/".format(os.path.basename(os.path.dirname(opt.model_path)))
    # save_attention_dir = save_dir + "attention/".format(os.path.basename(os.path.dirname(opt.model_path)))
    # save_csv_dir = save_dir + "csv/".format(os.path.basename(os.path.dirname(opt.model_path)))
    # os.makedirs(save_mp4_dir, exist_ok=True)
    # os.makedirs(save_attention_dir, exist_ok=True)
    # os.makedirs(save_csv_dir, exist_ok=True)
    # filename = re.sub(r'[\\/:*?"<>|]+','', input_text)[:20].replace(" ", "_") + '_focus=' + str(focus_word) + "_1"
    # generate(opt, text=input_text, lib=lib, model=model, focus_word=focus_word, filename=filename, tokenizer=tokenizer,
    #     save_csv_dir=save_csv_dir, save_mp4_dir=save_mp4_dir, save_attention_dir=save_attention_dir, TEXT_SPLIT=len(input_text.split()))
    # exit()




    # # ----------- Evaluation with Trinity Dataset ----------
    # text_dir = "C:/Users/b19.teshima/Documents/Gesture/Evaluations/Evaluation_data/Text"
    # audio_dir = "C:/Users/b19.teshima/Documents/Gesture/Evaluations/Evaluation_data/Audio"
    # save_dir = "C:/Users/b19.teshima/Documents/Gesture/Evaluations/Evaluation_data/Generated_Gesture/act2g_woattn_α=10.0_β=2.0_sep=8"
    # for textfile in tqdm(glob.glob(text_dir+"/*.txt")):
    #     with open(textfile, 'r') as f:
    #         text = f.readline()
        
    #     filename = os.path.basename(textfile)[:-4]
    #     input_audio = audio_dir + "/" + filename + '.wav'
    #     if not os.path.exists(input_audio):
    #         continue

    #     # if filename != "TestSeq006_8":
    #     #     continue

    #     save_mp4_dir = save_dir + "/stick_2d"
    #     save_attention_dir = save_dir + "/attention_log"
    #     save_csv_dir = save_dir + "/csv"
    #     os.makedirs(save_mp4_dir, exist_ok=True)
    #     os.makedirs(save_attention_dir, exist_ok=True)
    #     os.makedirs(save_csv_dir, exist_ok=True)

    #     if os.path.exists(save_csv_dir + '/' + filename + '.csv'):
    #         continue

    #     generate(opt, text=text, lib=lib, model=model, input_audio=input_audio, filename=filename, tokenizer=tokenizer,
    #       save_csv_dir=save_csv_dir, save_mp4_dir=save_mp4_dir, save_attention_dir=save_attention_dir)
    # exit()




    
    # ----------- Evaluation with TED Dataset ----------
    data_path = "./data/imagistic_gestures_clip.npy"
    data_dir = "Z:/Human/b19-teshima/TED_videos/VideoStorage/clip_poses"
    save_dir = "C:/Users/b19.teshima/Documents/Gesture/Evaluations/Evaluation_data/Generated_Gesture/act2g_ted_clipdata"
    valid_time = [2, 11] # 9sec~11sec
    fps = 25

    data = np.load(data_path, allow_pickle=True).item()

    gesture_lengths = np.array([len(text) for text in data['gesture']]) / fps
    indexes = np.where(gesture_lengths[np.where(gesture_lengths > valid_time[0])[0]] < valid_time[1])[0]

    for i in tqdm(indexes):

        text = data['text'][i]
        gesture_id = data['clip_id'][i]

        input_audio = data_dir + '/' + gesture_id[:11] + '/' + gesture_id + '/' + gesture_id + '.wav'
        if not os.path.exists(input_audio):
            continue

        save_mp4_dir = None
        save_attention_dir = None
        save_csv_dir = save_dir + "/act2g_csv"
        save_gt_dir = save_dir + "/gt_csv"
        # os.makedirs(save_mp4_dir, exist_ok=True)
        # os.makedirs(save_attention_dir, exist_ok=True)
        os.makedirs(save_csv_dir, exist_ok=True)
        os.makedirs(save_gt_dir, exist_ok=True)

        if os.path.exists(save_csv_dir + '/' + gesture_id + '.csv'):
            continue

        generate(opt, text=text, lib=lib, model=model, input_audio=input_audio, filename=gesture_id, 
                 save_csv_dir=save_csv_dir, save_mp4_dir=save_mp4_dir, save_attention_dir=save_attention_dir)
        
        # Save GT
        save_csv_path = save_gt_dir + '/' + gesture_id + ".csv"
        CMUPose2KinectData(data['gesture'][i], save_csv=save_csv_path, fps=25, isConvert=False)

        # Save audio
        save_audio_path = save_dir + "/wav/" + gesture_id + '.wav'
        os.makedirs(os.path.dirname(save_audio_path), exist_ok=True)
        shutil.copyfile(input_audio, save_audio_path)

        # Save Text
        save_text_path = save_dir + "/text/" + gesture_id + '.txt'
        os.makedirs(os.path.dirname(save_text_path), exist_ok=True)
        with open(save_text_path, "w") as f:
            f.write(text)

    exit()


    # ---------- Test ----------
    testdata_path = "./dataset/testdata_5-12keyframe.pkl"
    save_dir = "./images/test/{}/".format(os.path.basename(os.path.dirname(opt.model_path)))
    # testdata_path = "./dataset/data_5-12keyframe_1000gestures_BERT_distmat=vae.pkl"
    # save_dir = "./images/train/{}/".format(os.path.basename(os.path.dirname(opt.model_path)))
    
    test_data = pickle.load(open(testdata_path, "rb"))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'gesture/', exist_ok=True)
    os.makedirs(save_dir+'attention/', exist_ok=True)


    for i in range(len(test_data['texts'])):
        text = test_data['texts'][i]
        kf = len(test_data["keyframe_list"][i])
        filename = "{}_{}".format(str(i).zfill(4), re.sub(r'[\\/:*?"<>|]+','', text)[:20].replace(" ", "_"))
        save_mp4_dir = save_dir + "gesture"
        save_attention_dir = save_dir+"attention"

        if os.path.exists(save_mp4_dir + '/' + filename + '.mp4'):
            continue
        
        print("Input: ", text)
        generate(opt, text=text, lib=lib, model=model, filename=filename, save_mp4_dir=save_mp4_dir, save_attention_dir=save_attention_dir)


            
if __name__ == '__main__':
    main()

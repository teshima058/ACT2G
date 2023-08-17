# ACT2G: Attention-based Text-to-Gesture Generation

This repository contains the code of our paper

*ACT2G: Attention-based Contrastive Learning for Text-to-Gesture Generation (SCA 2023)*

You can download Youtube videos and transcripts, divide the videos into scenes, and extract human poses.
Please see the project page and paper for the details.  
 
<!-- [[Project page]](https://sites.google.com/view/youngwoo-yoon/projects/co-speech-gesture-generation) [[Paper]](https://arxiv.org/abs/1810.12541) -->


## Installation

The code is implemented in Python 3.7.1.  

```
# 1. Create a conda virtual environment
conda create -n act2g python=3.7.1 -y
conda activate act2g

# 2. Install Pytorch (https://pytorch.org/get-started/locally/)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 3. Install
pip install -r requirements.txt

```

## A step-by-step guide

1. Set config
   * Update paths and youtube developer key in `config.py` (the directories will be created if not exist).
   * Update target channel ID. The scripts are tested for TED and LaughFactory channels.

2. Execute `download_video.py`
   * Download youtube videos, metadata, and subtitles (./videos/*.mp4, *.json, *.vtt).

3. Execute `run_openpose.py`
   * Run [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract body, hand, and face skeletons for all vidoes (./skeleton/*.pickle). 

4. Execute `run_scenedetect.py`
   * Run [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/) to divide videos into scene clips (./clip/*.csv).
  
5. Execute `run_gentle.py`
   * Run [Gentle](https://github.com/lowerquality/gentle) for word-level alignments (./videos/*_align_results.json).
   * You should skip this step if you use auto-generated subtitles. This step is necessary for the TED Talks channel. 

6. Execute `run_clip_filtering.py`
   * Remove inappropriate clips.
   * Save clips with body skeletons (./clip/*.json).

7. *(optional)* Execute `review_filtered_clips.py`
   * Review filtering results.

8. Execute `make_ted_dataset.py`
   * Do some post processing and split into train, validation, and test sets (./script/*.pickle).


## Pre-built TED gesture dataset
 
Running whole data collection pipeline is complex and takes several days, so we provide the pre-built dataset for the videos in the TED channel.  

| | |
| --- | --- |
| Number of videos | 1,766 |
| Average length of videos | 12.7 min |
| Shots of interest | 35,685 (20.2 per video on average) |
| Ratio of shots of interest | 25% (35,685 / 144,302) |
| Total length of shots of interest | 106.1 h |

* [[ted_raw_poses.zip]](https://drive.google.com/open?id=1vvweoCFAARODSa5J5Ew6dpGdHFHoEia2) 
[[z01]](https://drive.google.com/open?id=1zR-GIx3vbqCMkvJ1HdCMjthUpj03XKwB) 
[[z02]](https://kaistackr-my.sharepoint.com/:u:/g/personal/zeroyy_kaist_ac_kr/EeAaPXuWXYNJk9AWTKZ30zEBR0hHnSuXEmetiOD412cZ7g?e=qVSeYk) 
[[z03]](https://drive.google.com/open?id=1uhfv6k0Q3E7bUIxYDAVjxKIjPM_gL8Wm)
[[z04]](https://drive.google.com/open?id=1VLi0oQBW8xetN7XmkGZ-S_KhD-DvbVQB)
[[z05]](https://drive.google.com/open?id=1F2wiRX421f3hiUkEeKcTBbtsgOEBy7lh) (split zip files, Google Drive or OneDrive links, total 80.9 GB)  
The result of Step 3. It contains the extracted human poses for all frames. 
* [[ted_shots_of_interest.zip, 13.3 GB]](https://drive.google.com/open?id=1kF7SVpxzhYEHCoSPpUt6aqSKvl9YaTEZ)  
The result of Step 6. It contains shot segmentation results ({video_id}.csv files) and shots of interest ({video_id}.json files). 
'clip_info' elements in JSON files have start/end frame numbers and a boolean value indicating shots of interest. 
The JSON files contain the extracted human poses for the shots of interest, 
so you don't need to download ted_raw_poses.zip unless the human poses for all frames are necessary.
* [[ted_gesture_dataset.zip, 1.1 GB]](https://drive.google.com/open?id=1lZfvufQ_CIy3d2GFU2dgqIVo1gdmG6Dh)  
The result of Step 8. Train/validation/test sets of speech-motion pairs. 
 
### Download videos and transcripts
We do not provide the videos and transcripts of TED talks due to copyright issues.
You should download actual videos and transcripts by yourself as follows:  
1. Download and copy [[video_ids.txt]](https://drive.google.com/open?id=1grFWC7GBIeF2zlaOEtCWw4YgqHe3AFU-) file which contains video ids into `./videos` directory.
2. Run `download_video.py`. It downloads the videos and transcripts in `video_ids.txt`.
Some videos may not match to the extracted poses that we provided if the videos are re-uploaded.
Please compare the numbers of frames, just in case.


## Citation 

If our code or dataset is helpful, please kindly cite the following paper:
```
@INPROCEEDINGS{
  yoonICRA19,
  title={Robots Learn Social Skills: End-to-End Learning of Co-Speech Gesture Generation for Humanoid Robots},
  author={Yoon, Youngwoo and Ko, Woo-Ri and Jang, Minsu and Lee, Jaeyeon and Kim, Jaehong and Lee, Geehyuk},
  booktitle={Proc. of The International Conference in Robotics and Automation (ICRA)},
  year={2019}
}
```
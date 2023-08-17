import os
import subprocess
import librosa
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np

def detectBeat(path_to_wavefile):
    wave, fs = librosa.load(path_to_wavefile)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(wave, sr=fs, hop_length=hop_length, aggregate=np.median)
    times = librosa.times_like(onset_env, sr=fs, hop_length=hop_length)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=fs)
    return onset_env, beats, times


if __name__ == "__main__":

    path_to_wavefile = "./.tmp/audio/company_creates_more.wav"
    save_dir = os.path.dirname(path_to_wavefile)

    onset_env, beats, times = detectBeat(path_to_wavefile)

    # Save Beat Figure
    plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    plt.legend()
    plt.show()
    plt.close()

    # Trim audio around beats
    sound = AudioSegment.from_file(path_to_wavefile, format="wav")
    for i,b in enumerate(beats):
        window_size = 0.3 
        start = int(max(times[b] - window_size, 0) * 1000)
        end = int((times[b] + window_size) * 1000)
        cut = sound[start:end]
        save_path = save_dir + "/" + os.path.basename(path_to_wavefile)[:-4] + "_" + str(i).zfill(4) + '.wav'
        cut.export(save_path, format='wav')

        # cmd = "ffmpeg -i {} -ss {} -t {} {}".format(path_to_wavefile, start, end, save_path)
        # subprocess.call(cmd)
 
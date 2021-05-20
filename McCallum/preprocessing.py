# -*- coding: utf-8 -*-

import os
import warnings
from pathlib import Path
import glob
import multiprocessing as mp
import numpy as np
import librosa
import madmom


def compute_dataset(i, n_bins=72, n_octave=6, min_freq=40, n_t=128):
    bpo = int(n_bins / n_octave)
    # Compute beat-centered CQTs for each track
    print("track n°" + str(i + 1) + "/" + str(len(paths)))
    print(paths[i])
    filename = Path(paths[i]).stem
    
    # skip processing if cqt file already exists
    if os.path.isfile(root_path + "cqts_personal/" + "cqts_" + str(filename) + ".npy"):
      print('already processed, return ...')
      return
    
    y, sr = librosa.load(paths[i])
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(paths[i])
    beats_time = np.array(proc(act))
    beats_idx = (beats_time * sr).astype(int)
    L = len(beats_idx)
    cqts = np.empty((L - 1, n_bins, n_t))
    for j in range(L - 1):
        n_samples = beats_idx[j + 1] - beats_idx[j]
        hop_size = int(np.ceil(n_samples / (n_t * 32)) * 32)  # hop_size must be multiple of 2^5 for 6 octaves CQT
        cqts_j = librosa.cqt(y[beats_idx[j]:int(beats_idx[j] + hop_size * (n_t - 1))], sr, hop_size, fmin=min_freq,
                             n_bins=n_bins, bins_per_octave=bpo)

        if cqts_j.shape[1] != n_t :
            print('warning : adapting CQT window')
            cqts_j = librosa.cqt(y[int(- hop_size * (n_t - 1)):], sr, hop_size, fmin=min_freq,
                             n_bins=n_bins, bins_per_octave=bpo)
        cqts[j] = cqts_j
    np.save(root_path + "beats_personal/" + "beats_" + str(filename), beats_time)
    np.save(root_path + "cqts_personal/" + "cqts_" + str(filename), cqts)


if __name__ == "__main__":
    root_path = "/ids-cluster-storage/storage/atiam-1005/music-structure-estimation/McCallum/"
    #data_path = "./../data/Isophonics/audio/*"
    #data_path = "/ids-cluster-storage/storage/atiam-1005/harmonixset/src/mp3s/*"
    data_path = "/ids-cluster-storage/storage/atiam-1005/music-structure-estimation/data/Personal/Top 2000/*"
    warnings.filterwarnings("ignore")
    paths = glob.glob(data_path)
    print(str(len(paths)) + " files to process ...")
    a_pool = mp.Pool()
    a_pool.map(compute_dataset, range(len(paths)))
    warnings.filterwarnings("always")
# -*- coding: utf-8 -*-

import os
import warnings
from pathlib import Path
import glob
import multiprocessing as mp
import numpy as np
import librosa
import madmom
import jams
import random

def compute_dataset(i, n_octave=6, bpo=12, min_freq=None, n_seg=16, h_l=32):
    

    
    n_bins = bpo * n_octave
    # Compute beat-centered CQTs for each track
    print("track n°" + str(i + 1) + "/" + str(len(paths)))
    filename = Path(paths[i]).stem
    print(filename)
    if os.path.isfile('{0}done/{1}.npy'.format(data_path, str(filename))):
        print('already processed, return ...')
        return
    # skip processing if cqt file already exists
    if os.path.isfile('{0}cqts/{1}.npy'.format(data_path, str(filename))):
        print('already processed, return ...')
        return
    np.save('{0}done/{1}.npy'.format(data_path, str(filename)), np.array([0]))
    # Read audio
    y, sr = librosa.load(paths[i])
    # Read segment annotations
    #track_annotations = jams.load('{0}references/{1}.jams'.format(data_path,str(filename)))
    #annotation_seg = track_annotations.search(namespace='segment_open')[0].data
    #boundaries = np.array([obs.time + 0.15 for obs in annotation_seg])
    #labels = [obs.value for obs in annotation_seg]
    
    with open('{0}references/segments/{1}.txt'.format(data_path,str(filename)), 'r') as f:
        data = f.readlines() # read raw lines into an array
        boundaries = []
        labels = []
        for raw_line in data:
            obs = raw_line.strip().split(' ')
            boundaries.append(float(obs[0]))
            labels.append(obs[1])
    
    
    # Estimate beat locations
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(paths[i])
    beats_time = np.array(proc(act))
    #beats = librosa.time_to_frames(beats_time, sr=sr, hop_length=h_l)
    # Associate segmentation labels with each beat
    beats_labels = label_beats(beats_time, boundaries, labels)
    beats = librosa.time_to_frames(beats_labels[:, 0], sr=sr, hop_length=h_l)
    n_beats = len(beats)
    
    # Compute CQT
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=h_l, fmin=min_freq, n_bins=n_bins, bins_per_octave=bpo))
    # Adapt beat frames to the CQT
    # Using feature clustering on the CQT to apply non-evenly-spaced subsegmentation of the beats
    sub_beats = librosa.segment.subsegment(C, beats, n_segments=n_seg)
    #print(sub_beats.shape)
    # Ensures sub-beats are well aligned with the original beats
    sub_beats= librosa.util.fix_frames(sub_beats, x_min=beats[0], x_max=beats[-1], pad=False)
    #print(sub_beats.shape)
    # Apply beat-synchronization to the CQT according to the sub-beats using median aggregation
    C = librosa.util.sync(C, sub_beats, aggregate=np.median, pad=False)
    #print(C.shape)
    #print(sub_beats.shape)
    #print(beats.shape)
    
    # Reshape the CQT so that one element corresponds to the CQT of an entire beat
    cqts = np.zeros((n_beats-1, n_bins, n_seg))
    for i in range(n_beats-1):
        cqts[i,: , :] = C[:, i*16:(i+1)*16]
    

    
    # Save CQTs, beats and labels
    np.save('{0}cqts/{1}.npy'.format(data_path, str(filename)), cqts)
    np.save('{0}beats_labels/{1}.npy'.format(data_path, str(filename)), beats_labels[:-1, :])
    
def label_beats(beats, boundaries, labels):
    idx_start = np.argmin(np.abs(beats - boundaries[1]))
    idx_end = np.argmin(np.abs(beats - boundaries[-1]))
    beats = beats[idx_start:(idx_end + 1)]
    # Transforming string labels into int labels
    boundaries = boundaries[1:-1]
    labels = labels[1:-1]
    labels_set = set(labels[1:-1])
    labels2id = {label : i for i, label in enumerate(labels)}
    
    # Applying labels to beats
    beats_labels = np.zeros((len(beats), 2))
    beats_labels[:, 0] = beats
    for i, t in enumerate(boundaries):
        idx_boundary = np.argmin(np.abs(beats - t))
        beats_labels[idx_boundary:, 1] = labels2id[labels[i]]

    return beats_labels

if __name__ == "__main__":
    data_path = "/tsi/clusterhome/atiam-1005/data/Harmonix/"
    warnings.filterwarnings("ignore")
    paths = glob.glob(data_path + 'audio/*')
    random.shuffle(paths)
    print(str(len(paths)) + " files to process ...")
    a_pool = mp.Pool()
    a_pool.map(compute_dataset, range(len(paths)))
    #for i in range(len(paths)):
    #    compute_dataset(i)
    warnings.filterwarnings("always")
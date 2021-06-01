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

def compute_dataset(i, n_bins=72, n_octave=6, min_freq=None, n_seg=16, h_l=64):
    bpo = int(n_bins / n_octave)
    # Compute beat-centered CQTs for each track
    print("track n°" + str(i + 1) + "/" + str(len(paths)))
    print(paths[i])
    filename = Path(paths[i]).stem
    
    # skip processing if cqt file already exists
    if os.path.isfile('{0}cqts/{1}.npy'.format(data_path, str(filename))) or os.path.isfile('{0}cqts/to_check/{1}.npy'.format(data_path, str(filename))):
      print('already processed, return ...')
      return

    # Read audio
    y, sr = librosa.load(paths[i])
    start_time = next((i for i, x in enumerate(y) if x), None)/sr
    end_time = len(y)/sr
    
    # Estimate beat locations
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(paths[i])
    beats_time = np.array(proc(act))
    beats = librosa.time_to_frames(beats_time, sr=sr, hop_length=h_l)
    n_beats = len(beats)
    # Compute CQT
    C = np.abs(librosa.cqt(y=y, sr=sr, hop_length=h_l, fmin=min_freq, n_bins=n_bins, bins_per_octave=bpo))
    # Adapt beat frames to the CQT
    beats = librosa.util.fix_frames(beats, x_max=C.shape[1], pad=False)
    beat_t = librosa.frames_to_time(beats, sr=sr, hop_length=h_l)
    # Using feature clustering on the CQT to apply non-evenly-spaced subsegmentation of the beats
    sub_beats = librosa.segment.subsegment(C, beats, n_segments=n_seg)
    # Ensures sub-beats are well aligned with the original beats
    sub_beats= librosa.util.fix_frames(sub_beats, x_min=beats[0], x_max=beats[-1], pad=False)
    
    # Apply beat-synchronization to the CQT according to the sub-beats using median aggregation
    C = librosa.util.sync(C, sub_beats, aggregate=np.median, pad=False)
    
    # Reshape the CQT so that one element corresponds to the CQT of an entire beat
    cqts = np.zeros((n_beats, n_bins, n_seg))
    for i in range(n_beats-1):
        cqts[i,: , :] = C[:, i*16:(i+1)*16]

    # Read segment annotations
    #track_annotations = jams.load('{0}references/jams/{1}.jams'.format(data_path,str(filename)))
    #annotation_seg = track_annotations.search(namespace='segment_open')[0].data
    
    # Associate segmentation labels with each beat
    #beats_labels, is_reliable = label_beats(beat_t, annotation_seg, start_time, end_time)
    
    # Check the reliability of the annotations and the subsegmentation
    #if not is_reliable:
    #    print('Preprocessing not reliable for {0}'.format(str(filename)))
    #    return
    
    # Save CQTs, beats and labels
    #np.save('{0}beats_labels/to_check/{1}.npy'.format(data_path, str(filename)), beats_labels)
    np.save('{0}cqts/{1}.npy'.format(data_path, str(filename)), cqts)

    
def label_beats(beats, annotation_seg, start_time, end_time):
    is_reliable = True
    times = np.array([obs.time for obs in annotation_seg])
    labels = [obs.value for obs in annotation_seg]
    # Transforming string labels into int labels
    labels.insert(0, 'silence')
    times = np.insert(times, 0, -1)
    labels_set = set(labels)
    labels2id = {label : i for i, label in enumerate(labels)}
    
    # Fitting track start_time with first non-silence label
    id_label_start = next((i for i, x in enumerate(labels) if x!='silence'))
    times = times + (start_time - times[id_label_start])
    # Checking the reliability of updated labels
    #if abs(times[-1] + annotation_seg[-1].duration - end_time) > 10:
    #    is_reliable = False
    
    # Applying labels to beats
    beats_labels = np.zeros((len(beats), 2))
    beats_labels[:, 0] = beats
    k = 0
    for i, b in enumerate(beats):
        if k < len(times)-1 and round(b, 1) >= round(times[k+1], 1):
            k +=1
        beats_labels[i,1] = labels2id[labels[k]]
    
    return beats_labels, is_reliable

if __name__ == "__main__":
    data_path = "/tsi/clusterhome/atiam-1005/data/Personal/"
    warnings.filterwarnings("ignore")
    paths = glob.glob(data_path + '100 Greatest Songs of the 80s/*')
    random.shuffle(paths)
    print(str(len(paths)) + " files to process ...")
    a_pool = mp.Pool()
    a_pool.map(compute_dataset, range(len(paths)))
    #for i in range(len(paths)):
    #  compute_dataset(i)
    warnings.filterwarnings("always")
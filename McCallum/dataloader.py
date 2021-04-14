from pathlib import Path
import os
import numpy as np
import multiprocessing as mp
import torch
import librosa
import random
import warnings
import jams
import glob
from torch.utils.data import DataLoader, Dataset
import madmom

class CQTsDataset(Dataset):
    """CQTs dataset."""

    def __init__(self, n_files, n_triplets, delta=(16, 16, 96), dim_cqt=(72, 128 * 4)):

        self.n_files = n_files
        self.delta = delta
        self.n_triplets = n_triplets
        self.dim_cqt = dim_cqt

    def __len__(self):
        return len(self.n_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        cqts = np.load(self.n_files[idx])
        dp, dnmin, dnmax = self.delta
        L = np.shape(cqts)[0]

        cqts_batch = torch.empty(self.n_triplets, 3, self.dim_cqt[0], self.dim_cqt[1])
        for j in range(self.n_triplets):
            a = np.random.randint(2, L - 2)
            p = np.random.randint(max(a - dp, 2), min(a + dp, L - 2))
            n1 = np.random.randint(max(a - dnmax, 2), max(a - dnmin, 3))
            n2 = np.random.randint(min(a + dnmin, L - 3), min(a + dnmax, L - 2))
            n = int(random.choice([n1, n2]))

            cqts_a = np.append(cqts[a - 2], cqts[a - 1], axis=1)
            cqts_a = np.append(cqts_a, cqts[a], axis=1)
            cqts_a = np.append(cqts_a, cqts[a + 1], axis=1)

            cqts_p = np.append(cqts[p - 2], cqts[p - 1], axis=1)
            cqts_p = np.append(cqts_p, cqts[p], axis=1)
            cqts_p = np.append(cqts_p, cqts[p + 1], axis=1)

            cqts_n = np.append(cqts[n - 2], cqts[n - 1], axis=1)
            cqts_n = np.append(cqts_n, cqts[n], axis=1)
            cqts_n = np.append(cqts_n, cqts[n + 1], axis=1)

            cqts_batch[j, 0] = torch.from_numpy(cqts_a)
            cqts_batch[j, 1] = torch.from_numpy(cqts_p)
            cqts_batch[j, 2] = torch.from_numpy(cqts_n)

        return cqts_batch


def compute_dataset(i, n_bins=72, n_octave=6, min_freq=40, n_t=128):
    bpo = int(n_bins / n_octave)
    # Compute beat-centered CQTs for each track
    # for i, tp in enumerate(paths):
    print("track n°" + str(i + 1) + "/" + str(len(paths)))
    print(paths[i])
    filename = Path(paths[i]).stem
    
    # skip processing if cqt file already exists
    if os.path.exists(root_path + "cqts_personal/" + "cqts_" + str(filename) + ".npy"):
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
    root_path = "./"
    #data_path = "./../data/Isophonics/audio/*"
    #data_path = "/ids-cluster-storage/storage/atiam-1005/harmonixset/src/mp3s/*"
    data_path = "/ids-cluster-storage/storage/atiam-1005/music-structure-estimation/data/Personal/Top 2000/*"
    warnings.filterwarnings("ignore")
    paths = glob.glob(data_path)
    print(str(len(paths)) + "files to process ...")
    a_pool = mp.Pool()
    a_pool.map(compute_dataset, range(len(paths)))

    warnings.filterwarnings("always")
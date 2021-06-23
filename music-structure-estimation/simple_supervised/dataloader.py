# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
import torch
from torch.utils.data import Dataset
from utils import label_beats
from pathlib import Path

class CQTsDataset(Dataset):
    """CQTs dataset."""

    def __init__(self, path_cqts, n_sampled=50):
        
        self.path_cqts = path_cqts
        self.n_sampled = n_sampled

    def __len__(self):
        return len(self.path_cqts)

    def __getitem__(self, idx):
        # Open numpy file of track i
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            cqts = np.load(self.path_cqts[idx])
            L, nf, nt = cqts.shape
        except ValueError:
            print("An error occured with file {}".format(self.path_cqts[idx]))
        
        
        # Normalization
        cqts = cqts / np.max(cqts)
        
        # Access the segmentation labels
        labels = np.load(self.path_cqts[idx].replace('cqts', 'beats_labels'))[:, 1]
        assert L == len(labels)
        
        # return n_sampled cqts
        id_sampled = (torch.randperm(L-3) + 2)[:self.n_sampled].numpy()
        cqts_batch = torch.zeros((self.n_sampled, nf, 4 * nt))
        for k, id_s in enumerate(id_sampled):
            cqts_batch[k] = torch.from_numpy(self.append_cqts(cqts, id_s - 2, id_s + 2))      
        
        return cqts_batch, labels[id_sampled]
            
            
        return a, p, n
    
    def append_cqts(self, cqts, low, high):
        cqts_app = np.append(cqts[low], cqts[low + 1], axis=1)
        for i in range(low + 2, high):
            cqts_app = np.append(cqts_app, cqts[i], axis= 1)
        return cqts_app

# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CQTsDataset(Dataset):
    """CQTs dataset."""

    def __init__(self, path_cqts, normalization='max'):   
        self.path_cqts = path_cqts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
        self.normalization = normalization
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
        if self.normalization == 'max':
            cqts = cqts / np.max(cqts)
        elif self.normalization == 'log_max_centered':
            cqts = np.log(cqts + 5e-3)
            cqts = cqts - np.mean(cqts)
            cqts = cqts / np.max(np.abs(cqts))
        
        # Access the segmentation labels
        labels = np.load(self.path_cqts[idx].replace('cqts', 'beats_labels'))[:, 1].astype(np.int64)
        assert L == len(labels)
        
        # compute cqt inputs
        cqts_track = torch.zeros((L, nf, 4 * nt)).to(self.device)
        for k in range(L):
            cqts_track[k] = torch.from_numpy(self.append_cqts(cqts, k - 2, k + 2, nt))      
        
        # compute ssm input
        one_hot_labels = F.one_hot(torch.from_numpy(labels))
        ssm = one_hot_labels.matmul(one_hot_labels.transpose(-1, -2)).to(self.device)

        
        return cqts_track.float(), ssm.float()

    
    def append_cqts(self, cqts, low, high, nt):
        N, F, T = np.shape(cqts)
        cqts_app = np.zeros((F, (high - low) * T))
        for i, pos  in enumerate(range(low, high)):
            if 0 <= pos < N:
                cqts_app[:, i*nt:(i+1)*nt] = cqts[pos]        
        return cqts_app  

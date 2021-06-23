# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset


class CQTsDataset(Dataset):
    """CQTs dataset."""

    def __init__(self, path_cqts):   
        self.path_cqts = path_cqts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "error")

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
        
        
        cqts = cqts / np.max(cqts)
        
        # Access the segmentation labels
        labels = np.load(self.path_cqts[idx].replace('cqts', 'beats_labels'))[:, 1]
        assert L == len(labels)
        
        # compute cqt inputs
        cqts_track = torch.zeros((L - 4, nf, 4 * nt)).to(self.device)
        for k in range(L - 4):
            cqts_track[k] = torch.from_numpy(self.append_cqts(cqts, k, k + 4))      
        
        # compute ssm input
        ssm = torch.zeros((L - 4, L - 4)).to(self.device)
        
        for i in range(L - 4):
            for j in range(L - 4):
                ssm[i, j] = torch.from_numpy(np.array(labels[i + 2] != labels[j + 2]))
        
        return cqts_track, ssm

    
    def append_cqts(self, cqts, low, high):
        cqts_app = np.append(cqts[low], cqts[low + 1], axis=1)
        for i in range(low + 2, high):
            cqts_app = np.append(cqts_app, cqts[i], axis= 1)
        return cqts_app

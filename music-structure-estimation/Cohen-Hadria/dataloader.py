# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SSMsDataset(Dataset):
    """CQTs dataset."""

    def __init__(self, path_ssm, n_sampled=16):   
        self.path_ssm = path_ssm
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
        self.n_sampled = n_sampled

    def __len__(self):
        return len(self.path_cqts)

    def __getitem__(self, idx):
        # Open numpy file of track i
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            smm_packed = np.load(self.path_ssm[idx])
            L = ssm_packed.shape[0]
        except ValueError:
            print("An error occured with file {}".format(self.path_cqts[idx]))
        
        # Access the boundaries groundtruth
        boundaries = np.load(self.path_ssm[idx].replace('ssm', 'boundaries')).astype(np.int64)
        assert L == len(boundaries)
        
        
        # Select positive and negative examples
        idx_positive = np.where(boundaries)[0]
        idx_negative = np.where(1 - boundaries)[0]
        Lp = len(idx_positive)
        idx_positive = np.random.choice(idx_positive, min(Lp, self.n_sampled//2), replace=False)
        idx_negative = np.random.choice(idx_negative, max(self.n_sampled - Lp, self.n_sampled//2), replace=False)
        
        ssm_packed = torch.from_numpy(ssm_packed[np.append(idx_positive, idx_negative)])
        boundaries = torch.from_numpy(boundaries[np.append(idx_positive, idx_negative)]) 
        
        return ssm_packed.float().to(device), boundaries.float().to(device)

    
    def append_cqts(self, cqts, low, high, nt):
        N, F, T = np.shape(cqts)
        cqts_app = np.zeros((F, (high - low) * T))
        for i, pos  in enumerate(range(low, high)):
            if 0 <= pos < N:
                cqts_app[:, i*nt:(i+1)*nt] = cqts[pos]        
        return cqts_app  

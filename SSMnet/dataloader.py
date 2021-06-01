# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
import torch
from torch.utils.data import Dataset
from utils import label_beats
from pathlib import Path

class CQTsDataset(Dataset):
    """CQTs dataset."""

    def __init__(self, path_cqts, path_labels, path_beats, bias=True, n_sampled=16, dim_embed=128):
        
        self.path_cqts = path_cqts
        self.path_labels = path_labels
        self.delta = delta
        self.bias = True
        self.n_sampled = n_sampled
        self.dim_embed = dim_embed

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
        
        # Access the segmentation labels
        filename = Path(self.path_cqts[idx]).stem.replace('cqts_','')
        labels = label_beats('{0}{1}.txt'.format(self.path_labels, filename), '{0}beats_{1}.npy'.format())
        assert L == len(labels)
        
        # return n_sampled cqts
        id_sampled = (torch.randperm(L-3) + 2)[:self.n_sampled].numpy()
        cqts_batch = torch.zeros((self.n_sampled, nf, nt))
        for k, id_s in enumerate(id_sampled):
            cqts_batch[k] = torch.from_numpy(self.append_cqts(cqts, id_s - 2, id_s + 2))      
        
        return cqts_batch, labels[id_sampled]
        
        
        return cqts_batch
        
    def sample_triplet(self, cqts):
        dp, dnmin, dnmax = self.delta
        L = np.shape(cqts)[0]
        a = torch.randint(2, L - 1, (1,)).item()
        p_inf = max(a - dp, 2)
        nl_inf = max(a - dnmax, 2)
        nr_inf = min(a + dnmin, L - 2)
        p_sup = min(a + dp, L - 2)
        nl_sup = max(a - dnmin, 3)
        nr_sup = min(a + dnmax, L - 1)
        
        if self.bias and 19 <= a <= L - 20:
            sample_left = self.compare_segments(cqts, a)
            (p_inf, p_sup) = (a + 1, p_sup) if sample_left else (p_inf, a)
        elif a in [2, L - 3]:
            sample_left = (a != 2)
        else:
            sample_left = torch.randint(2, (1,)).item()
        
        
        p, n = a, a
        while a == p or a == n:
            p = torch.randint(p_inf, p_sup, (1,)).item()
            n = torch.randint(nl_inf, nl_sup, (1,)).item() if sample_left \
                else torch.randint(nr_inf, nr_sup, (1,)).item()
            
            
        return a, p, n
    
    def append_cqts(self, cqts, low, high):
        cqts_app = np.append(cqts[low], cqts[low + 1], axis=1)
        for i in range(low + 2, high):
            cqts_app = np.append(cqts_app, cqts[i], axis= 1)
        return cqts_app

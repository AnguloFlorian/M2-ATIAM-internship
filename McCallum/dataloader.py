# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
import torch
from torch.utils.data import Dataset

class CQTsDataset(Dataset):
    """CQTs dataset."""

    def __init__(self, n_files, n_triplets=16, bias=True, delta=(16, 1, 96), dim_cqt=(72, 64)):

        self.n_files = n_files
        self.delta = delta
        self.bias = bias
        self.n_triplets = n_triplets
        self.dim_cqt = dim_cqt

    def __len__(self):
        return len(self.n_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            cqts = np.load(self.n_files[idx])
        except ValueError:
            print("An error occured with file {}".format(self.n_files[idx]))

        cqts_batch = torch.empty(self.n_triplets, 3, self.dim_cqt[0], self.dim_cqt[1])
        for j in range(self.n_triplets):
            
            a, p, n = self.sample_triplet(cqts)
            cqts_batch[j, 0] = torch.from_numpy(self.append_cqts(cqts, a - 2, a + 2))
            cqts_batch[j, 1] = torch.from_numpy(self.append_cqts(cqts, p - 2, p + 2))
            cqts_batch[j, 2] = torch.from_numpy(self.append_cqts(cqts, n - 2, n + 2))

        return cqts_batch
        
    def sample_triplet(self, cqts):
        dp, dnmin, dnmax = self.delta
        L = np.shape(cqts)[0]
        a = torch.randint(2, L - 1, (1,)).item()
        p_inf = max(a - dp, 2)
        nl_inf = max(a - dnmax, 2)
        nr_inf = min(a + dnmin, L - 2)
        p_sup = min(a + dp, L - 1)
        nl_sup = max(a - dnmin + 1, 3)
        nr_sup = min(a + dnmax, L - 1)
        
        if self.bias and 19 <= a <= L - 20:
            sample_left = self.compare_segments(cqts, a)
            (p_inf, p_sup) = (a + 1, p_sup) if sample_left else (p_inf, a)
        if a in [2, L - 2]:
            sample_left = (a != 2)
        else:
            sample_left = torch.randint(2, (1,)).item()
        
        
        p, n = a, a
        while (a == p or a == n):
            p = torch.randint(p_inf, p_sup, (1,)).item()
            n = torch.randint(nl_inf, nl_sup, (1,)).item() if sample_left \
                else torch.randint(nr_inf, nr_sup, (1,)).item()
            
            
        return a, p, n
    
    def append_cqts(self, cqts, low, high):
        cqts_app = np.append(cqts[low], cqts[low + 1], axis=1)
        for i in range(low + 2, high):
            cqts_app = np.append(cqts_app, cqts[i], axis= 1)
        return cqts_app
    
    def compare_segments(self, cqts, a):
        eps = 1e-7j
        cqts_left1 = np.log(np.abs(np.fft.fft2(np.log(np.abs(self.append_cqts(cqts, a - 20, a - 12)) + eps))) + eps)
        cqts_left2 = np.log(np.abs(np.fft.fft2(np.log(np.abs(self.append_cqts(cqts, a - 8, a)) + eps))) + eps)
        cqts_right1 = np.log(np.abs(np.fft.fft2(np.log(np.abs(self.append_cqts(cqts, a, a + 8)) + eps))) + eps)
        cqts_right2 = np.log(np.abs(np.fft.fft2(np.log(np.abs(self.append_cqts(cqts, a + 12, a + 20)) + eps))) + eps)
        
        return norm(cqts_left2 - cqts_left1, 2) > norm(cqts_right2 - cqts_right1, 2)


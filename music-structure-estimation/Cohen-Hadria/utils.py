# -*- coding: utf-8 -*-
import numpy as np

def append_cqts(cqts, low, high):
    N, F, T = np.shape(cqts)
    cqts_app = np.zeros((F, (high - low) * T))
    for i, pos  in enumerate(range(low, high)):
        if 0 <= pos < N:
            cqts_app[:, i*T:(i+1)*T] = cqts[pos]        
    return cqts_app  
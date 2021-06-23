import torch
from bisect import bisect
import numpy as np
import math

def triplet_loss(a, p, n, device, alpha = 0.1):
    # inputs :
    #   - a : anchor audio embeddings
    #   - p : positive example audio embeddings
    #   - n : negative example audio embeddings
    #   - alpha : parameter of the triplet loss (error margin)
    # output :
    #   - triplet_loss computed with the L2-norm
    
    loss = 0
    zero = torch.FloatTensor([0]).to(device)

    for i in range(a.size(0)):
        loss += torch.max(zero, torch.norm(a[i] - p[i])**2 - torch.norm(a[i] - n[i])**2 + alpha)
  
    return loss



def update_stats(n_anchors, fp_vec, fn_matrix, boundaries, duration, delta_p, delta_n):
  
    anchors = np.random.uniform(0, duration, (n_anchors))
    for a in anchors:
    # update false positive vector
        for i, dp  in enumerate(delta_p):
            p = np.random.uniform(max(a - dp, 0), min(a + dp, duration))
            fp_vec[i] += (bisect(boundaries ,a) != bisect(boundaries, p))

        for i in range(len(delta_n)):
            dnmin = delta_n[i]
            for j  in range(i + 1, len(delta_n)):
                dnmax = delta_n[j]
                n1 = np.random.uniform(max(a - dnmax, 0), max(a - dnmin, 0))
                n2 = np.random.uniform(min(a + dnmin, duration), min(a + dnmax, duration))
                n = random.choice([n1,n2])
                # update false negative matrix
                fn_matrix[i, j] += (bisect(boundaries, a) == bisect(boundaries, n))

    return fp_vec, fn_matrix
    
    

def label_beats(path_beats, path_labels):
    # Associate a music structure analysis label to every beat according to the annotation
    beats = np.load(path_beats)
    f = open(path_labels,'r')
    segments = [['silence', -1]]
    for line in f.readlines():
        line = line.split()
        segments.append([truncate(float(line[0]), 1), line[1]])

    cursor = 0
    labels = [''] * len(beats)
    k = 0
    for i, b in enumerate(beats):
        if k < len(segments)-1 and b >= segments[k+1][0]:
            k +=1
        labels[i] = segments[k][1]
    return labels


def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor
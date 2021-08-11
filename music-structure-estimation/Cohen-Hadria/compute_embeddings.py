# -*- coding: utf-8 -*-
import torch
import numpy as np
import glob
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from model import ConvNet
from utils import append_cqts

data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/cqts/*"
files = glob.glob(data_path_harmonix)
files.extend(glob.glob(data_path_isoph))


model = ConvNet()
eps = 1e-8
F = 8

for f in files:
    cqts = np.load(f)
    cqts = cqts / np.max(cqts + 1e-7)
    L, nf, nt = cqts.shape

    
    # Compute the SSM from the CQT
    ssm_cqts = cosine_similarity(cqts.reshape((L, -1)))
    # Compute the SSM from the embeddings and the 2D-FMC
    cqts_track = torch.zeros(L, nf, 4 * nt)
    fmc2d = np.zeros((L, nf, 8 * nt))
    for k in range(L):
        cqts_track[k] = torch.from_numpy(append_cqts(cqts, k - 2, k + 2))
        fmc2d[k] = np.log(np.abs(np.fft.fft2(np.log(np.abs(append_cqts(cqts, k - 4, k + 4)) + eps))) + eps)
        
    embeds = model.apply_cnn(cqts_track)
    ssm_embeds = torch.matmul(embeds, embeds.transpose(-2, -1))
    ssm_2dfmc = cosine_similarity(fmc2d.reshape((L, -1)))
    
    ssm_cqts = np.pad(ssm_cqts, (F//2, F//2), 'wrap')
    ssm_embeds = np.pad(ssm_embeds.detach().cpu().numpy(), (F//2, F//2), 'wrap')
    ssm_2dfmc = np.pad(ssm_2dfmc, (F//2, F//2), 'wrap')
    
    # building the subSSMs
    ssm_packed = np.zeros((L, 3, F, F))
    for k in range(L):
        ssm_packed[k, 0] = ssm_cqts[k:k+F, k:k+F]
        ssm_packed[k, 1] = ssm_embeds[k:k+F, k:k+F]
        ssm_packed[k, 2] = ssm_2dfmc[k:k+F, k:k+F]        
    
    np.save(f.replace('cqts', 'ssm'), ssm_packed)
    
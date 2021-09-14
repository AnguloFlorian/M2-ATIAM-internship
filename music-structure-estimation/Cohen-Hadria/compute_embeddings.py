# -*- coding: utf-8 -*-
import torch
import numpy as np
import glob
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from model import ConvNet
from utils import append_cqts
import multiprocessing as mp
import random

def compute_ssms(id_f):
    f = files[id_f] 
    print(f)
    eps = 1e-8
    cqts = np.load(f)
    cqts = cqts / np.max(cqts + eps)
    L, nf, nt = cqts.shape
    
    """
    cqts_flat = torch.from_numpy(cqts.reshape(L, -1))
    ssm_cqts = torch.cdist(cqts_flat, cqts_flat)
    ssm_cqts = (1 - ssm_cqts/torch.max(ssm_cqts))
    """
    
    
    try:
        ssm_packed = np.load(f.replace('cqts', 'ssm/McCallum'), allow_pickle=False)
        #ssm_packed[1] = ssm_cqts.numpy()
        #np.save(f.replace('cqts', 'ssm'), ssm_packed, allow_pickle=False)
        #return
    except ValueError:
        try:
            ssm_packed = np.load(f.replace('cqts', 'ssm/McCallum'), allow_pickle=True)
            #ssm_packed[1] = ssm_cqts.numpy()
            #np.save(f.replace('cqts', 'ssm'), ssm_packed, allow_pickle=False)
            #return
        except OSError:
            print("ALERTE", f)
            return
            

    # Compute the SSM from the embeddings and the 2D-FMC
    cqts_track = torch.zeros(L, nf, 4 * nt)
    fmc2d = np.zeros((L, nf, 8 * nt))
    for k in range(L):
        cqts_track[k] = torch.from_numpy(append_cqts(cqts, k - 2, k + 2))

        #fmc2d[k] = np.log(np.abs(np.fft.fft2(np.log(np.abs(append_cqts(cqts, k - 4, k + 4)) + eps))) + eps)

    
    """
    fmc2d_flat = torch.from_numpy(fmc2d).reshape(L, -1)
    ssm_fmc2d = torch.cdist(fmc2d_flat, fmc2d_flat)
    ssm_fmc2d = 1 - ssm_fmc2d/torch.max(ssm_fmc2d)    
    """
    
    embeds = model.apply_cnn(cqts_track.unsqueeze(1))
    #np.save(f.replace('cqts', 'embeds/SSMnet'), embeds.detach().numpy())
    ssm_embeds = torch.cdist(embeds, embeds)
    ssm_embeds = 1 - ssm_embeds/torch.max(ssm_embeds)    

    
    #ssm_cqts = ssm_cqts.numpy()
    ssm_embeds = ssm_embeds.detach().numpy()
    #ssm_fmc2d = ssm_fmc2d.numpy()

    
    # building the subSSMs
    #ssm_packed = np.zeros((3, L, L))
    #for k in range(L):
    ssm_packed[0] = ssm_embeds
    #ssm_packed[1] = ssm_cqts
    #ssm_packed[2] = ssm_fmc2d        
    
    np.save(f.replace('cqts', 'ssm/SSMnet'), ssm_packed, allow_pickle=False)



if __name__ == "__main__":
    data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/cqts/*.npy"
    data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/cqts/*.npy"
    data_path_beatles = "/tsi/clusterhome/atiam-1005/data/Isophonics/cqts/BeatlesTUT/*.npy"
    root_path = "/tsi/clusterhome/atiam-1005/M2-ATIAM-internship/music-structure-estimation/Cohen-Hadria/"
    files = glob.glob(data_path_harmonix)
    files.extend(glob.glob(data_path_beatles))
    files.extend(glob.glob(data_path_isoph))
    model = ConvNet()
    model.load_state_dict(torch.load('{0}weights/best_ssmnet.pt'.format(root_path)), strict=False)
    random.shuffle(files)
    print(str(len(files)) + " files to process ...")
    a_pool = mp.Pool()
    a_pool.map(compute_ssms, range(len(files)))
    #for i in range(len(files)):
    #    compute_ssms(i)

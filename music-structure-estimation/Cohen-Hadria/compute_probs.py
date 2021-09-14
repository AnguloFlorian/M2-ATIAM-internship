# -*- coding: utf-8 -*-

import random
import glob
import torch
from pathlib import Path
from model import CohenNet
import numpy as np

print('libraries imported')

device = torch.device("cuda:0" if torch.cuda.is_available() else "error")
root_path = "/tsi/clusterhome/atiam-1005/M2-ATIAM-internship/music-structure-estimation/Cohen-Hadria/"
data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/ssm/McCallum/*.npy"
data_path_beatles = "/tsi/clusterhome/atiam-1005/data/Isophonics/ssm/McCallum/BeatlesTUT/*.npy"


files = glob.glob(data_path_beatles)

model = CohenNet().to(device)

model.load_state_dict(torch.load('{0}weights/final_exp_McCallum_best.pt'.format(root_path)), strict=False)

padding = 7

for f in files:
    print(f)
    ssm = np.load(f)[0]
    L = ssm.shape[0]
    F = padding * 2 + 1
    ssm = torch.from_numpy(np.pad(ssm, (padding, padding), 'wrap')).to(device)
    sub_ssms = torch.zeros(L, F, F).to(device)
    for k in range(L):
        range_subssm = range(k, k+F)
        sub_ssms[k] = ssm[range_subssm, range_subssm]
    
    probs = model(sub_ssms)
    filename = Path(f).stem
    output = probs.view(-1).cpu().detach().numpy()
    np.save("./probs/" + filename, output)
    

# -*- coding: utf-8 -*-
import numpy as np
import glob

data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/beats_labels/*"
data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/beats_labels/*"

files = glob.glob(data_path_harmonix)
files.extend(glob.glob(data_path_isoph))


for f in files:
    beats_labels = np.load(f)
    L = beats_labels.shape[0]
    boundaries = np.zeros((L,))
    boundaries[1:] = (beats_labels[:-1, 1] != beats_labels[1:,1]) 
    
    
    np.save(f.replace('beats_labels', 'boundaries'), boundaries)
    
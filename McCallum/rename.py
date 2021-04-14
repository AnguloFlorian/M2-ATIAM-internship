from pathlib import Path
import os
import numpy as np
import glob


save_path = "./beats_harmonix2/"
data_path = "./beats_harmonix/*"
paths = glob.glob(data_path)

for _, p in enumerate(paths):
    beats = np.load(p)
    filename = str(Path(p).stem).replace("beats_harmonix","")
    np.save(save_path + filename, beats)
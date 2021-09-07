# Simple MSAF example
from __future__ import print_function
import msaf
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import median_filter as med_f
import glob

def compute_metrics(f):
    print(f)
    # open ssm file
    ssm_embeds = np.load(f)[0]
    # open beat locations file
    beats = np.load(f.replace('ssm','beats_labels'))[:, 0]
    # open groundtruth boundaries file
    annot_boundaries = np.load(f.replace('ssm', 'boundaries'))
    annot_boundaries = np.where(annot_boundaries)[0]
    
    # Estimate boundaries
    est_boundaries = estimate_boundaries(ssm_embeds)
    
    # Convert boundaries from index to time
    annot_boundaries_t = beats[annot_boundaries.astype('int')]
    est_boundaries_t = beats[est_boundaries.astype('int')]
    
    # Convert boundaries from single events to intervals
    ann_inter, est_inter = convert_boundaries(annot_boundaries_t, est_boundaries_t, beats)
    
    # Evaluate estimated boundaries
    metrics_eval = msaf.eval.compute_results(ann_inter, est_inter, ann_labels=None, est_labels=None, bins=251, est_file=root_path+"result")
    
    return metrics_eval
    # 1. Select audio file
    #audio_file = "./audio/0004_abc.mp3"
    
    # 2. Segment the file using the default MSAF parameters (this might take a few seconds)
    #boundaries, labels = msaf.process(audio_file, boundaries_id="olda")
    #print('Estimated boundaries:', boundaries)
    
    # 3. Save segments using the MIREX format
    #out_file = 'segments.txt'
    #print('Saving output to %s' % out_file)
    #msaf.io.write_mirex(boundaries, labels, out_file)
    
    # 4. Evaluate the results
    #evals = msaf.eval.process(audio_file)
    #print(evals)
    
# novel function
def novelty(ssm, nu, g):
    eta = 0
    for i in range(-kappa, kappa+1):
        for j in range(-kappa, kappa+1):
            eta += ssm[nu + i, nu + j] * g[i, j]
    return eta
    
def estimate_boundaries(ssm):
    L = len(ssm)
    # Pad ssm to account for the checkerboard kernel size and peak_to_mean window size
    ssm_checkerboard = np.pad(ssm,(kappa + T, kappa + T), 'wrap')
    
    # Compute novelty
    nov = np.empty((L + 2*T))
    for i in range(L+2*T):
        nov[i] = novelty(ssm_checkerboard, i + kappa, g)
    
    # Compute peak_to_mean ratio
    peak_to_mean = np.zeros((L))
    for i in range(T, L + T):
        peak_to_mean[i - T] = (2*T+1)*nov[i]/np.sum(nov[i-T:i+T+1])
        
    # Find peaks above threshold tau
    peaks, _ = find_peaks(peak_to_mean[4:], distance=4)
    above_treshold = np.where(peak_to_mean>=tau)[0]
    est_boundaries = np.intersect1d(peaks, above_treshold)
    
    return est_boundaries


def convert_boundaries(annot, estim, beats_t):
    start = beats_t[0]
    end = beats_t[-1]+0.5
    L_ann = len(annot)
    L_est = len(estim)
    
    ann_inter = np.zeros((L_ann + 1, 2))
    est_inter = np.zeros((L_est + 1, 2))
    
    ann_inter[0, 0] = start
    est_inter[0, 0] = start
    
    for i in range(L_ann):
        ann_inter[i,1] = annot[i]
        ann_inter[i + 1, 0] = annot[i]
    
    for i in range(L_est):
        est_inter[i,1] = estim[i]
        est_inter[i + 1, 0] = estim[i]       
    
    ann_inter[-1, 1] = end
    est_inter[-1, 1] = end
    
    return ann_inter, est_inter
    
if __name__ == "__main__":
    # constant parameters
    kappa = 32
    sigma = 18.5
    T = 8
    tau = 1.1
    
    # chessboard kernel
    chb_size = 2*kappa+1
    g = np.zeros((chb_size, chb_size))
    for i in range(-kappa, kappa+1):
        for j in range(-kappa, kappa+1):
            if (abs (i) == 0 or abs(j) == 0):
                pass
            else:
                r2 = i**2+j**2
                g[i+kappa,j+kappa] = np.sign(i) * np.sign(j) * np.exp(-r2/(2*sigma**2))

    data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/ssm/*.npy"
    data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/ssm/*.npy"
    data_path_beatles = "/tsi/clusterhome/atiam-1005/data/Isophonics/ssm/BeatlesTUT/*.npy"
    root_path = "/tsi/clusterhome/atiam-1005/M2-ATIAM-internship/music-structure-estimation/Evaluation/"
    
    name_exp = "McCallum_Isoph"
    files = glob.glob(data_path_isoph)
    L = len(files)
    print(str(len(files)) + " files to process ...")
    metrics = np.zeros((8, len(files)))
    for k in range(len(files)):
       # compute metric for file i
       metrics_file = compute_metrics(files[k])
       
       # assign each metric to the output vector
       
       # F-measure of hit rate at 3 seconds
       metrics[0, k] = metrics_file["HitRate_3F"]
       # Precision of hit rate at 3 seconds
       metrics[1, k] = metrics_file["HitRate_3P"]
       # Recall of hit rate at 3 seconds
       metrics[2, k] = metrics_file["HitRate_3R"]
       # F-measure of hit rate at 0.5 seconds
       metrics[3, k] = metrics_file["HitRate_0.5F"]
       # Precision of hit rate at 0.5 seconds
       metrics[4, k] = metrics_file["HitRate_0.5P"]
       # Recall of hit rate at 0.5 seconds
       metrics[5, k] = metrics_file["HitRate_0.5R"]
       # F-measure of hit rate at 3 seconds weighted
       metrics[6, k] = metrics_file["HitRate_w3F"]
       # F-measure of hit rate at 0.5 seconds weighted
       metrics[7, k] = metrics_file["HitRate_w0.5F"]
    
    
    np.save(root_path + name_exp, metrics)
       
      

# Simple MSAF example
from __future__ import print_function
import msaf
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import median_filter as med_f
import glob
from hmmlearn import hmm
import warnings

def compute_metrics(f):
    print(f)
    # open ssm file
    embeds = np.load(f)
    # open beat locations and annotated labels  file
    beats_labels = np.load(f.replace("embeds/" + model_name,"beats_labels"))
    beats = beats_labels[:, 0]
    ann_labels = beats_labels[:, 1]
    # get number of classes
    num_classes = len(set(ann_labels))
    
    # open groundtruth boundaries file
    annot_boundaries = np.load(f.replace("embeds/" + model_name, "boundaries"))
    annot_boundaries = np.insert(np.where(annot_boundaries)[0], 0, int(0))
    
    # Estimate labels sequence
    est_labels = estimate_labels(embeds, num_classes)
    #est_labels = post_process(est_labels)
    
    
    # Deduce estimated boundaries from estimated labels
    # est_boundaries = np.zeros_like(est_labels)
    #est_boundaries[1:] = (est_labels[:-1] != est_labels[1:]) 
    #est_boundaries = np.insert(np.where(est_boundaries)[0], 0, int(0))
    est_boundaries = annot_boundaries
    
    # Convert labels sequence into segment label assignments
    est_labels = convert_from_annot(est_labels, annot_boundaries)
    #est_labels = est_labels[est_boundaries.astype('int')]
    ann_labels = ann_labels[annot_boundaries.astype('int')]
    
    # Convert boundaries from index to time
    annot_boundaries_t = beats[annot_boundaries.astype('int')]
    est_boundaries_t = beats[est_boundaries.astype('int')]
    

    # Convert boundaries from single events to intervals
    ann_inter, est_inter = convert_boundaries(annot_boundaries_t, est_boundaries_t, beats)
    
    print(est_labels)    
    # Evaluate estimated boundaries
    warnings.filterwarnings("ignore")
    metrics_eval = msaf.eval.compute_results(ann_inter, est_inter, ann_labels, est_labels, bins=251, est_file=root_path+"result")
    warnings.filterwarnings("default")
    return metrics_eval



def convert_boundaries(annot, estim, beats_t):
    end = beats_t[-1]+0.5
    L_ann = len(annot)
    L_est = len(estim)
    
    ann_inter = np.zeros((L_ann, 2))
    est_inter = np.zeros((L_est, 2))
    
    for i in range(L_ann-1):
        ann_inter[i, 0] = annot[i]
        ann_inter[i, 1] = annot[i + 1]
    
    for i in range(L_est-1):
        est_inter[i, 0] = estim[i]
        est_inter[i, 1] = estim[i + 1]       
    
    ann_inter[-1, 0] = annot[-1]
    est_inter[-1, 0] = estim[-1]
    ann_inter[-1, 1] = end
    est_inter[-1, 1] = end
    
    return ann_inter, est_inter
    
    
def estimate_labels(embeddings, num_classes):
    num_classes = max(2, num_classes)
    remodel = hmm.GaussianHMM(n_components=num_classes, covariance_type="full", n_iter=100, tol=0.01)
    remodel.fit(embeddings)
    est_labels = remodel.predict(embeddings)
    return est_labels
    
    
def post_process(est_labels):
    est_labels[0] = est_labels[1]
    L = len(est_labels)
    
    pos = 0
    while pos < L:
        i = pos
        while i + 1 < L and est_labels[i + 1] == est_labels[i]:
            i += 1 
        if i - pos < 20 and  i + 1 < L:
            if pos > 0:
                est_labels[pos:(i+1)] = est_labels[pos-1]
            else:
                est_labels[pos:(i+1)] = est_labels[i+1]                
        pos = i + 1
        
    return est_labels
            

def convert_from_annot(est_labels, annot):

    L = len(annot)
    convert_est_labels = np.zeros((L))
    for i in range(L-1):
         convert_est_labels[i] = np.bincount(est_labels[annot[i]:annot[i+1]]).argmax()
    convert_est_labels[-1] = np.bincount(est_labels[annot[-1]:]).argmax()
    
    return convert_est_labels
    
if __name__ == "__main__":
    #model_name = "McCallum"
    model_name = "SSMnet"
    data_path_harmonix = "/tsi/clusterhome/atiam-1005/data/Harmonix/embeds/" + model_name + "/*.npy"
    data_path_isoph = "/tsi/clusterhome/atiam-1005/data/Isophonics/embeds/" + model_name + "/*.npy"
    data_path_beatles = "/tsi/clusterhome/atiam-1005/data/Isophonics/embeds/" + model_name + "/BeatlesTUT/*.npy"
    root_path = "/tsi/clusterhome/atiam-1005/M2-ATIAM-internship/music-structure-estimation/Evaluation/"
    
    name_exp = model_name + "_Beatles_labels"
    files = glob.glob(data_path_beatles)
    L = len(files)
    print(str(len(files)) + " files to process ...")
    metrics = np.zeros((14, len(files)))
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
       # F-measure of pair-wise frame clustering
       metrics[8, k] = metrics_file["PWF"]
       # Precision of pair-wise frame clustering
       metrics[9, k] = metrics_file["PWP"]
       # Recall of pair-wise frame clustering
       metrics[10, k] = metrics_file["PWR"]
       # F-measure normalized entropy score
       metrics[11, k] = metrics_file["Sf"]
       # Oversegmentation normalized entropy score
       metrics[12, k] = metrics_file["So"]
       # Undersegmentation normalized entropy score
       metrics[13, k] = metrics_file["Su"]
    np.save(root_path + name_exp, metrics)
       
      

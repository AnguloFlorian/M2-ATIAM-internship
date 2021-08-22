import torch
from torch import linalg as LA

def weighted_bce_loss(pred, target, lp_ord=0.5, coeff=0.4):
    N = target.shape[-1]
    num_elem = N**2#(N-1)/2
    num_non_zeros = target.sum()# - N)/2
    #pred = torch.triu(pred, diagonal=1)
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)   
    #target = torch.triu(target, diagonal=1)     

    
    weights = [num_non_zeros/num_elem, (num_elem - num_non_zeros)/num_elem]
    zero_contrib = weights[0] * (1 - target) * (1 - pred).log()
    one_contrib = weights[1] * target * pred.log()
    
    loss = - torch.mean(zero_contrib + one_contrib)

    return  loss
import torch
from torch import linalg as LA

def weighted_bce_loss(pred, target, pred_labels=None, lp_ord=0.5, coeff=0.4):
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)                 
    num_non_zeros = target.sum()
    num_elem = target.shape[-1] ** 2
    weights = [num_non_zeros/num_elem, (num_elem - num_non_zeros)/num_elem]
    zero_contrib = weights[0] * (1 - target) * (1 - pred).log()
    one_contrib = weights[1] * target * pred.log()
    
    loss = - torch.mean(zero_contrib + one_contrib)
    
    if pred_labels is not None:
        lp_norm = torch.mean(torch.norm(pred_labels, p=lp_ord, dim=-1))
    else:
        lp_norm = 0
    return loss + coeff * lp_norm, loss
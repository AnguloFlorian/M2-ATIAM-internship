import torch

def weighted_bce_loss(pred, target):
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)                 
    num_non_zeros = target.sum()
    num_elem = target.shape[-1] ** 2
    weights = [num_non_zeros/num_elem, (num_elem - num_non_zeros)/num_elem]
    zero_contrib = weights[0] * (1 - target) * (1 - pred).log()
    one_contrib = weights[1] * target * pred.log()
    return - torch.mean(zero_contrib + one_contrib)
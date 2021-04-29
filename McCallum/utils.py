import torch
from bisect import bisect



def triplet_loss(a, p, n, device, alpha = 0.1):
    # inputs :
    #   - a : anchor audio embeddings
    #   - p : positive example audio embeddings
    #   - n : negative example audio embeddings
    #   - alpha : parameter of the triplet loss (error margin)
    # output :
    #   - triplet_loss computed with the L2-norm
    
    loss = 0
    zero = torch.FloatTensor([0]).to(device)

    for i in range(a.size(0)):
        loss += torch.max(zero, torch.norm(a[i] - p[i])**2 - torch.norm(a[i] - n[i])**2 + alpha)
  
    return loss



def update_stats(n_anchors, fp_vec, fn_matrix, boundaries, duration, delta_p, delta_n):
  
    anchors = np.random.uniform(0, duration, (n_anchors))
    for a in anchors:
    # update false positive vector
        for i, dp  in enumerate(delta_p):
            p = np.random.uniform(max(a - dp, 0), min(a + dp, duration))
            fp_vec[i] += (bisect(boundaries ,a) != bisect(boundaries, p))

        for i in range(len(delta_n)):
            dnmin = delta_n[i]
            for j  in range(i + 1, len(delta_n)):
                dnmax = delta_n[j]
                n1 = np.random.uniform(max(a - dnmax, 0), max(a - dnmin, 0))
                n2 = np.random.uniform(min(a + dnmin, duration), min(a + dnmax, duration))
                n = random.choice([n1,n2])
                # update false negative matrix
                fn_matrix[i, j] += (bisect(boundaries, a) == bisect(boundaries, n))

    return fp_vec, fn_matrix
    
    

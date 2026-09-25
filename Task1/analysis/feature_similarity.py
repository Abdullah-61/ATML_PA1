import torch
import torch.nn.functional as F

def compute_cosine_stability(clean_feats, trans_feats):
    """
    Computes representation cosine stability:
    I_T = (1/N) * sum( (f(x)^T f(T(x))) / (||f(x)||_2 * ||f(T(x))||_2) )
    clean_feats: [N, D]
    trans_feats: [N, D]
    """
    sims = F.cosine_similarity(clean_feats, trans_feats, dim=-1)
    return sims.mean().item()

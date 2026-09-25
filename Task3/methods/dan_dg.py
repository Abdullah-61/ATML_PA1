import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    def __init__(self, bandwidth_multipliers=(0.5, 1.0, 2.0)):
        super().__init__()
        self.multipliers = bandwidth_multipliers

    def compute_pairwise_dist_sq(self, x, y):
        n = x.size(0)
        m = y.size(0)
        x_norm = (x ** 2).sum(dim=1, keepdim=True).expand(n, m)
        y_norm = (y ** 2).sum(dim=1, keepdim=True).expand(m, n).t()
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist_sq, min=0.0)

    def forward(self, source_features, target_features):
        b_s = source_features.size(0)
        b_t = target_features.size(0)
        combined = torch.cat([source_features, target_features], dim=0)

        dist_sq_total = self.compute_pairwise_dist_sq(combined, combined)
        triu_mask = torch.triu(torch.ones_like(dist_sq_total, dtype=torch.bool), diagonal=1)
        pairwise_vals = dist_sq_total[triu_mask]

        median_dist_sq = torch.median(pairwise_vals)
        if median_dist_sq.item() <= 0:
            median_dist_sq = torch.tensor(1.0, device=source_features.device)

        dist_ss = self.compute_pairwise_dist_sq(source_features, source_features)
        dist_tt = self.compute_pairwise_dist_sq(target_features, target_features)
        dist_st = self.compute_pairwise_dist_sq(source_features, target_features)

        k_ss = torch.zeros_like(dist_ss)
        k_tt = torch.zeros_like(dist_tt)
        k_st = torch.zeros_like(dist_st)

        for mult in self.multipliers:
            bandwidth = 2.0 * (mult * median_dist_sq)
            k_ss = k_ss + torch.exp(-dist_ss / bandwidth)
            k_tt = k_tt + torch.exp(-dist_tt / bandwidth)
            k_st = k_st + torch.exp(-dist_st / bandwidth)

        loss_ss = (k_ss.sum() - torch.diagonal(k_ss).sum()) / (b_s * (b_s - 1))
        loss_tt = (k_tt.sum() - torch.diagonal(k_tt).sum()) / (b_t * (b_t - 1))
        loss_st = (2.0 * k_st.sum()) / (b_s * b_t)

        mmd_loss = loss_ss + loss_tt - loss_st
        return torch.clamp(mmd_loss, min=0.0)

class DANDGLoss(nn.Module):
    def __init__(self, lambda_mmd=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mmd = MMDLoss(bandwidth_multipliers=(0.5, 1.0, 2.0))
        self.lambda_mmd = lambda_mmd

    def forward(self, source_logits, source_labels, feats_by_domain):
        loss_cls = self.ce(source_logits, source_labels)
        
        domains = list(feats_by_domain.keys())
        mmd_penalties = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                f_i = feats_by_domain[domains[i]]
                f_j = feats_by_domain[domains[j]]
                mmd_penalties.append(self.mmd(f_i, f_j))
                
        loss_mmd = torch.stack(mmd_penalties).mean()
        total_loss = loss_cls + self.lambda_mmd * loss_mmd
        return total_loss, {
            "loss_cls": loss_cls.item(),
            "loss_mmd": loss_mmd.item(),
            "total_loss": total_loss.item()
        }

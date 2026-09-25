import torch
import torch.nn as nn
import torch.nn.functional as F
from task2.models.domain_discriminator import ReverseLayerF

def multilinear_conditioning(features, logits):
    """
    Computes g(x) = vec(f (x) p), where p = softmax(logits).
    features: [B, D] (512)
    logits: [B, C] (7)
    Returns: [B, D * C] (3584)
    """
    probs = F.softmax(logits, dim=1)
    b, d = features.size()
    c = probs.size(1)
    # Batch outer product
    g = torch.bmm(features.unsqueeze(2), probs.unsqueeze(1))
    return g.view(b, d * c)

class ConditionalDomainDiscriminator(nn.Module):
    def __init__(self, in_features=3584, hidden_dim=256, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x, alpha=1.0):
        rev_x = ReverseLayerF.apply(x, alpha)
        return self.net(rev_x)

class CDANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_cls = nn.CrossEntropyLoss()
        self.ce_domain = nn.CrossEntropyLoss()

    def forward(self, source_logits, source_labels, source_dom_logits, target_dom_logits):
        loss_cls = self.ce_cls(source_logits, source_labels)

        d_source_labels = torch.zeros(source_dom_logits.size(0), dtype=torch.long, device=source_dom_logits.device)
        d_target_labels = torch.ones(target_dom_logits.size(0), dtype=torch.long, device=target_dom_logits.device)

        loss_d_s = self.ce_domain(source_dom_logits, d_source_labels)
        loss_d_t = self.ce_domain(target_dom_logits, d_target_labels)
        loss_domain = 0.5 * (loss_d_s + loss_d_t)

        total_loss = loss_cls + loss_domain
        return total_loss, {
            "loss_cls": loss_cls.item(),
            "loss_domain": loss_domain.item(),
            "total_loss": total_loss.item()
        }

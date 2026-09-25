import torch
import torch.nn as nn

class DANNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_cls = nn.CrossEntropyLoss()
        self.ce_domain = nn.CrossEntropyLoss()

    def forward(self, source_logits, source_labels, source_domain_logits, target_domain_logits):
        loss_cls = self.ce_cls(source_logits, source_labels)

        # Source domain label = 0, Target domain label = 1
        d_source_labels = torch.zeros(source_domain_logits.size(0), dtype=torch.long, device=source_domain_logits.device)
        d_target_labels = torch.ones(target_domain_logits.size(0), dtype=torch.long, device=target_domain_logits.device)

        loss_d_s = self.ce_domain(source_domain_logits, d_source_labels)
        loss_d_t = self.ce_domain(target_domain_logits, d_target_labels)
        loss_domain = 0.5 * (loss_d_s + loss_d_t)

        total_loss = loss_cls + loss_domain
        return total_loss, {
            "loss_cls": loss_cls.item(),
            "loss_domain": loss_domain.item(),
            "total_loss": total_loss.item()
        }

import torch
import torch.nn as nn
import torch.nn.functional as F

class PROSERModel(nn.Module):
    def __init__(self, base_model, num_known=10, num_dummies=5):
        super().__init__()
        self.base_model = base_model
        self.num_known = num_known
        self.num_dummies = num_dummies

        # Retain original known fc, attach randomly initialized dummy classifiers
        self.dummy_fc = nn.Linear(512, num_dummies)
        nn.init.kaiming_normal_(self.dummy_fc.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.dummy_fc.bias, 0)

    def forward_pre_layer2(self, x):
        return self.base_model.forward_pre_layer2(x)

    def forward_post_layer2(self, h):
        out = self.base_model.layer3(h)
        out = self.base_model.layer4(out)
        out = self.base_model.avgpool(out)
        feat = torch.flatten(out, 1)
        logits_known = self.base_model.fc(feat)
        logits_dummy = self.dummy_fc(feat)
        logits_all = torch.cat([logits_known, logits_dummy], dim=1)
        return logits_all, feat

    def forward(self, x):
        feat = self.base_model.forward_features(x)
        logits_known = self.base_model.fc(feat)
        logits_dummy = self.dummy_fc(feat)
        logits_all = torch.cat([logits_known, logits_dummy], dim=1)
        return logits_all, feat

class PROSERLoss(nn.Module):
    def __init__(self, num_known=10, num_dummies=5, beta=1.0, gamma=0.1):
        super().__init__()
        self.num_known = num_known
        self.num_dummies = num_dummies
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits_cls, labels_cls, logits_data=None):
        # 1. Standard classification on known classes
        loss_ce = F.cross_entropy(logits_cls[:, :self.num_known], labels_cls)

        # 2. Classifier-placeholder loss:
        # Mask ground-truth class, encourage dummy classifiers to dominate remaining logits
        batch_size = logits_cls.size(0)
        masked_logits = logits_cls.clone()
        masked_logits[torch.arange(batch_size), labels_cls] = -1e9
        prob_masked = F.softmax(masked_logits, dim=1)
        dummy_prob_sum = prob_masked[:, self.num_known:].sum(dim=1).clamp(min=1e-8)
        loss_cp = -torch.log(dummy_prob_sum).mean()

        total_loss = loss_ce + self.beta * loss_cp

        # 3. Data-placeholder loss (manifold mixup representations trained to dummy classifiers)
        loss_dp = torch.tensor(0.0, device=logits_cls.device)
        if logits_data is not None and logits_data.size(0) > 0:
            prob_data = F.softmax(logits_data, dim=1)
            dummy_prob_data = prob_data[:, self.num_known:].sum(dim=1).clamp(min=1e-8)
            loss_dp = -torch.log(dummy_prob_data).mean()
            total_loss = total_loss + self.gamma * loss_dp

        return total_loss, {
            "loss_ce": loss_ce.item(),
            "loss_cp": loss_cp.item(),
            "loss_dp": loss_dp.item(),
            "total_loss": total_loss.item()
        }

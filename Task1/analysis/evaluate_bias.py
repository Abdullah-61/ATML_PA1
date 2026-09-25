import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from task1.configs.config import SEED

@torch.no_grad()
def extract_all_backbones(dataset, backbones, norms, device, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    feats_r50, feats_vit, feats_clip, all_labels = [], [], [], []
    use_amp = torch.cuda.is_available()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        all_labels.append(labels)

        with torch.amp.autocast("cuda", enabled=use_amp):
            f_r = backbones["resnet50"](norms["resnet50"](imgs))
            f_v = backbones["vit_b16"](norms["vit_b16"](imgs))
            f_c = backbones["clip"].encode_image(norms["clip"](imgs))
            # Explicit normalized CLIP embedding per prompt specification
            f_c = f_c / f_c.norm(dim=-1, keepdim=True)

        feats_r50.append(f_r.float().cpu())
        feats_vit.append(f_v.float().cpu())
        feats_clip.append(f_c.float().cpu())

    targets = torch.cat(all_labels, dim=0)
    return {
        "resnet50": (torch.cat(feats_r50, dim=0), targets),
        "vit_b16": (torch.cat(feats_vit, dim=0), targets),
        "clip": (torch.cat(feats_clip, dim=0), targets),
    }

def train_linear_head(in_dim, train_f, train_y, val_f, val_y, device, seed=SEED):
    torch.manual_seed(seed)
    head = nn.Linear(in_dim, 10).to(device)
    nn.init.normal_(head.weight, std=0.1)
    nn.init.zeros_(head.bias)

    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(list(zip(train_f, train_y)), batch_size=64, shuffle=True)

    best_val_acc = -1.0
    best_weights = None
    no_improve_epochs = 0

    for epoch in range(50):
        head.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = head(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            v_logits = head(val_f.to(device))
            v_preds = v_logits.argmax(dim=1).cpu()
            val_acc = accuracy_score(val_y, v_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = head.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= 5:
                break

    head.load_state_dict(best_weights)
    head.eval()
    return head

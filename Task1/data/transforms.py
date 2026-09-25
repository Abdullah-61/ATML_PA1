import torch
from torchvision.transforms import functional as TF

def apply_grayscale(img_tensor):
    """
    Applies ITU-R BT.601 grayscale and replicates across 3 channels.
    img_tensor: [C, H, W] in range [0, 1]
    """
    return TF.rgb_to_grayscale(img_tensor, num_output_channels=3)

def apply_hue_rotation_180(img_tensor):
    """
    Rotates hue by +180 degrees (0.5 in PyTorch range [-0.5, 0.5]).
    Preserves luminance and saturation while altering chromatic polarity.
    """
    return TF.adjust_hue(img_tensor, hue_factor=0.5)

def apply_translation(img_tensor, shift_x, shift_y):
    """
    Translates image using reflection padding followed by a shifted crop.
    img_tensor: [C, H, W]
    """
    c, h, w = img_tensor.shape
    pad = max(abs(shift_x), abs(shift_y))
    if pad == 0:
        return img_tensor
    padded = TF.pad(img_tensor, padding=pad, padding_mode="reflect")
    return TF.crop(padded, top=pad - shift_y, left=pad - shift_x, height=h, width=w)

def apply_patch_shuffle_4x4(img_tensor, generator):
    """
    Divides image into 4x4 grid (16 patches) and applies non-identity permutation.
    """
    c, h, w = img_tensor.shape
    grid_size = 4
    ph, pw = h // grid_size, w // grid_size

    patches = []
    for r in range(grid_size):
        for col in range(grid_size):
            patches.append(img_tensor[:, r * ph:(r + 1) * ph, col * pw:(col + 1) * pw])

    perm = torch.randperm(16, generator=generator)
    while torch.equal(perm, torch.arange(16)):
        perm = torch.randperm(16, generator=generator)

    shuffled_patches = [patches[i] for i in perm]
    rows = []
    for r in range(grid_size):
        row = torch.cat(shuffled_patches[r * grid_size:(r + 1) * grid_size], dim=2)
        rows.append(row)
    return torch.cat(rows, dim=1)

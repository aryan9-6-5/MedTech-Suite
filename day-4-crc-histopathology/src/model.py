"""ConvNeXt-Tiny classifier and GPU augmentations (dihedral, and optional stain / colour jitter)."""
import math

import torch
import torch.nn as nn
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(n_classes=9, pretrained=True):
    w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.convnext_tiny(weights=w)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    return m


def normalise(x):
    """x in [0, 1], (B, 3, H, W)."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def dihedral(x):
    """Random flips and transpose per sample: the 8 symmetries of the square. Histology has no canonical orientation."""
    n = x.size(0)
    dev = x.device
    for dim in (2, 3):
        m = (torch.rand(n, device=dev) < 0.5).view(-1, 1, 1, 1)
        x = torch.where(m, x.flip(dim), x)
    m = (torch.rand(n, device=dev) < 0.5).view(-1, 1, 1, 1)
    return torch.where(m, x.transpose(2, 3), x)


# Stain vectors (Ruifrok & Johnston) as used by scikit-image: rows are haematoxylin, eosin, DAB in RGB optical density.
_RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
_RGB_FROM_HED = _RGB_FROM_HED / _RGB_FROM_HED.norm(dim=1, keepdim=True)
_HED_FROM_RGB = torch.linalg.inv(_RGB_FROM_HED)


def stain_jitter(x, sigma_alpha=0.1, sigma_beta=0.08):
    """Perturb each sample's haematoxylin / eosin / DAB optical densities (Tellez et al. 2018). x in [0, 1]."""
    n = x.size(0)
    dev = x.device
    od = -torch.log(x.clamp(1 / 255, 1.0))                       # (B, 3, H, W)
    od = od.permute(0, 2, 3, 1)                                   # (B, H, W, 3)
    hed = od @ _HED_FROM_RGB.to(dev)
    alpha = 1 + (torch.rand(n, 1, 1, 3, device=dev) * 2 - 1) * sigma_alpha
    beta = (torch.rand(n, 1, 1, 3, device=dev) * 2 - 1) * sigma_beta
    od2 = (hed * alpha + beta) @ _RGB_FROM_HED.to(dev)
    return torch.exp(-od2).clamp(0, 1).permute(0, 3, 1, 2)


def colour_jitter(x, brightness=0.15, contrast=0.15):
    n = x.size(0)
    dev = x.device
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * brightness
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * contrast
    mu = x.mean((1, 2, 3), keepdim=True)
    return ((x - mu) * c + mu).mul(b).clamp(0, 1)


def augment(x, arm):
    """arm 'A': geometry only. arm 'B': geometry plus stain and colour jitter (the pre-registered intervention)."""
    x = dihedral(x)
    if arm == "B":
        x = colour_jitter(stain_jitter(x))
    return x

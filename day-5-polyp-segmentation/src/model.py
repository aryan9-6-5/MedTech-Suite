"""U-Net with a swappable ImageNet-pretrained encoder (ResNet-34 or ConvNeXt-Tiny) and one shared decoder, plus GPU augmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Net(nn.Module):
    """Encoder gives 4 feature maps at strides 4, 8, 16, 32. The decoder is identical for both encoders."""

    def __init__(self, enc, pretrained=True):
        super().__init__()
        if enc == "resnet34":
            r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            stages = [nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4]
            ch = [64, 128, 256, 512]
        elif enc == "convnext":
            w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            f = torchvision.models.convnext_tiny(weights=w).features
            stages = [f[0:2], f[2:4], f[4:6], f[6:8]]
            ch = [96, 192, 384, 768]
        else:
            raise ValueError(enc)
        self.stages = nn.ModuleList(stages)
        dch = [64, 128, 256]
        self.dec = nn.ModuleList([
            _block(ch[3] + ch[2], dch[2]),            # stride 32 -> 16
            _block(dch[2] + ch[1], dch[1]),           # 16 -> 8
            _block(dch[1] + ch[0], dch[0]),           # 8 -> 4
        ])
        self.head = nn.Sequential(nn.Conv2d(dch[0], 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        x = feats[3]
        for blk, skip in zip(self.dec, (feats[2], feats[1], feats[0])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skip], 1))
        return F.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)   # logits, (B, 1, H, W)


def normalise(x):
    """x in [0, 1], (B, 3, H, W)."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x, m):
    """Same for both encoders. Per-sample flips and transpose applied to image and mask together, then brightness / contrast on the image."""
    n, dev = x.size(0), x.device
    z = torch.cat([x, m], 1)
    for dim in (2, 3):
        k = (torch.rand(n, device=dev) < 0.5).view(-1, 1, 1, 1)
        z = torch.where(k, z.flip(dim), z)
    k = (torch.rand(n, device=dev) < 0.5).view(-1, 1, 1, 1)
    z = torch.where(k, z.transpose(2, 3), z)
    x, m = z[:, :3], z[:, 3:]
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    mu = x.mean((1, 2, 3), keepdim=True)
    return ((x - mu) * c + mu).mul(b).clamp(0, 1), m

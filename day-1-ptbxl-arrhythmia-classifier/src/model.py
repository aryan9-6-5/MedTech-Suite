import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride, k=15, p=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p)
        self.skip = (
            nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
            if (cin != cout or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        h = torch.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(self.drop(h)))
        return torch.relu(h + self.skip(x))


class ECGResNet(nn.Module):
    """1D ResNet over (batch, 12 leads, 1000 samples) -> 5 superclass logits."""

    def __init__(self, n_leads=12, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 32, 15, 1, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU()
        )
        cfg = [(32, 64, 2), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1)]
        self.blocks = nn.Sequential(*[Block(a, b, s) for a, b, s in cfg])
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)).mean(-1))

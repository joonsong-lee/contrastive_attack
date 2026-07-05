"""Basic building blocks for NullSwap (ConvBlock, SEBlock, SEResBlock, DeconvBlock)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU, with optional MaxPool afterwards. Paper spec (Section III-B)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if use_pool else None

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class SEBlock(nn.Module):
    """Standard Squeeze-and-Excitation (Hu et al. 2018)."""

    def __init__(self, ch, reduction=16):
        super().__init__()
        hid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, hid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hid, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEResBlock(nn.Module):
    """ResNet bottleneck (1x1 reduce → 3x3 → 1x1 expand) with SE before the shortcut add.
    Paper states 'ResNet bottleneck + SENet'. Exact channel widths not specified; we use
    the common bottleneck = out_ch // 4."""

    def __init__(self, in_ch, out_ch, stride=1, reduction=16):
        super().__init__()
        bot = max(out_ch // 4, 8)
        self.conv1 = nn.Conv2d(in_ch, bot, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bot)
        self.conv2 = nn.Conv2d(bot, bot, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bot)
        self.conv3 = nn.Conv2d(bot, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        r = self.shortcut(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        y = self.se(y)
        return self.relu(y + r)


class DeconvBlock(nn.Module):
    """ConvTranspose2d stride=2 → BN → ReLU. 2× spatial upsample."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

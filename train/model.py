"""Compact fully-convolutional density regressor for point-supervised counting.

The network downsamples the input image by a factor of 8 and regresses a
single-channel density map. The predicted vehicle count is the sum over that
map, which is the standard paradigm introduced by CSRNet.
"""
from __future__ import annotations

from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 convs + BN + ReLU (VGG-style), no downsampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _dilated_block(in_ch: int, out_ch: int, dilation: int) -> nn.Sequential:
    """Single dilated 3x3 conv + BN + ReLU to widen receptive field at low res."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CountNet(nn.Module):
    """CSRNet-inspired counting network. Output is at 1/8 input resolution.

    Args:
        in_channels: image channels (3 for RGB).
        base: channel multiplier; channels grow base -> 4*base across 3 stages.
    """

    def __init__(self, in_channels: int = 3, base: int = 64) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(_conv_block(in_channels, base), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(_conv_block(base, base * 2), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(_conv_block(base * 2, base * 4), nn.MaxPool2d(2))

        channels = base * 4
        self.backend = nn.Sequential(
            _dilated_block(channels, channels, 1),
            _dilated_block(channels, channels, 2),
            _dilated_block(channels, channels, 4),
            _dilated_block(channels, channels, 2),
        )
        # Linear output, no activation: a ReLU here dies the moment the head's
        # pre-activation turns negative, freezing the head at zero with no
        # gradient to recover. CSRNet likewise uses a bare 1x1 conv as its head.
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.backend(x)
        return self.head(x)

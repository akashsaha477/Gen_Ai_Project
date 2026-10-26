"""Network pieces shared by the generator families: strided convolution stacks and an upsampling decoder.

Both work for any beat length (63, 180 or 360 samples): the decoder makes ``8 * ceil(L / 8)`` samples
with three x2 upsamplings and crops to ``L``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

SLOPE = 0.2  # LeakyReLU slope


def down_stack(in_channels: int, channels: tuple[int, ...] | list[int], kernel: int) -> nn.Sequential:
    """Stride-2 convolutions with LeakyReLU; no normalisation (a WGAN-GP critic must not use batch norm)."""
    layers: list[nn.Module] = []
    for out_channels in channels:
        layers += [nn.Conv1d(in_channels, out_channels, kernel, stride=2, padding=kernel // 2), nn.LeakyReLU(SLOPE)]
        in_channels = out_channels
    return nn.Sequential(*layers)


def flat_size(module: nn.Module, in_channels: int, n_samples: int) -> int:
    """Number of features ``module`` produces for one input of ``in_channels`` x ``n_samples``."""
    with torch.no_grad():
        return int(module(torch.zeros(1, in_channels, n_samples)).numel())


class UpsampleDecoder(nn.Module):
    """Vector to beat: a linear layer to a short sequence, then x2 upsampling and convolution per entry
    of ``channels``, a final convolution to one channel, and a crop to ``n_samples``."""

    def __init__(self, in_dim: int, n_samples: int, channels: tuple[int, ...] | list[int], kernel: int) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.base = math.ceil(n_samples / 2 ** len(channels))
        self.fc = nn.Linear(in_dim, channels[0] * self.base)
        blocks: list[nn.Module] = []
        previous = channels[0]
        for out_channels in channels:
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(previous, out_channels, kernel, padding=kernel // 2),
                nn.LeakyReLU(SLOPE),
            ]
            previous = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv1d(previous, 1, kernel, padding=kernel // 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc(h), SLOPE).view(h.shape[0], -1, self.base)
        return self.out(self.blocks(x))[:, 0, : self.n_samples]

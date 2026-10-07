"""Probe CNN: a small, fixed classifier used only to compare Module 1 input representations."""

from __future__ import annotations

import torch
from torch import nn


class ProbeCNN(nn.Module):
    """Three conv blocks, global average pooling, linear head, optional RR side input.

    Works for any window length that survives three poolings (63, 180 and 360 samples all do).
    The side branch standardises its inputs with batch statistics learned from training batches only.
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_side: int = 0,
        channels: tuple[int, ...] = (16, 32, 64),
        kernel: int = 7,
        side_hidden: int = 16,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.n_side = n_side
        if n_side:
            self.side = nn.Sequential(nn.BatchNorm1d(n_side), nn.Linear(n_side, side_hidden), nn.ReLU())
            self.head = nn.Linear(in_channels + side_hidden, n_classes)
        else:
            self.side = None
            self.head = nn.Linear(in_channels, n_classes)

    def forward(self, x: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` is ``[batch, length]`` or ``[batch, 1, length]``; ``side`` is ``[batch, n_side]``."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        pooled = self.features(x).mean(dim=-1)
        if self.side is not None:
            if side is None:
                raise ValueError("This probe was built with a side input; pass side features")
            pooled = torch.cat([pooled, self.side(side)], dim=1)
        elif side is not None:
            raise ValueError("This probe has no side input")
        return self.head(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

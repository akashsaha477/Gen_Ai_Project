"""The four phase2_baseline_classifiers architectures (PLAN Phase 2).

| variant | class | definition |
|---|---|---|
| A | ``ShallowCNN`` | 3 conv blocks (conv-BN-ReLU-pool), global average pooling, linear head |
| B | ``ResNet1D`` | stem convolution, 4 residual stages, global average pooling, linear head |
| C | ``BiLSTMClassifier`` | 2-layer bidirectional LSTM; last states and mean-pooled outputs concatenated; linear head |
| D | ``CNNBiLSTM`` | 2 conv blocks, then 1 bidirectional LSTM layer; last and mean-pooled states; linear head |

Every model takes the beat ``x`` ``[batch, length]`` and, if built with ``n_side > 0``, the RR side
features ``[batch, n_side]`` (standardised by a batch-norm layer, as in the probe). Every model keeps
its last layer in ``head`` (what weight transfer leaves out), exposes ``embed(x, side)`` (the
penultimate features: the feature hook), and the convolutional ones expose ``last_conv`` (for Grad-CAM).
"""

from __future__ import annotations

import torch
from torch import nn


class SideInput(nn.Module):
    """RR side features -> a small dense representation, standardised with batch statistics."""

    def __init__(self, n_side: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(n_side), nn.Linear(n_side, hidden), nn.ReLU())

    def forward(self, side: torch.Tensor) -> torch.Tensor:
        return self.net(side)


class _Classifier(nn.Module):
    """Shared plumbing: optional side input, head, and the checks on how the model is called."""

    def _setup_head(self, n_features: int, n_classes: int, n_side: int, side_hidden: int) -> None:
        self.n_side = n_side
        self.side = SideInput(n_side, side_hidden) if n_side else None
        self.head = nn.Linear(n_features + (side_hidden if n_side else 0), n_classes)

    def _with_side(self, features: torch.Tensor, side: torch.Tensor | None) -> torch.Tensor:
        if self.side is not None:
            if side is None:
                raise ValueError("This model was built with a side input; pass side features")
            return torch.cat([features, self.side(side)], dim=1)
        if side is not None:
            raise ValueError("This model has no side input")
        return features

    def waveform_features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - overridden
        raise NotImplementedError

    def embed(self, x: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        """Penultimate features: what the head sees."""
        if x.dim() == 3:
            x = x[:, 0, :]
        return self._with_side(self.waveform_features(x), side)

    def forward(self, x: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x, side))


def conv_block(in_channels: int, out_channels: int, kernel: int) -> list[nn.Module]:
    return [nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2), nn.BatchNorm1d(out_channels), nn.ReLU(), nn.MaxPool1d(2)]


class ShallowCNN(_Classifier):
    def __init__(self, n_classes: int = 4, n_side: int = 0, channels: tuple[int, ...] = (32, 64, 128), kernel: int = 7, side_hidden: int = 16) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = 1
        for out_channels in channels:
            layers += conv_block(previous, out_channels, kernel)
            previous = out_channels
        self.features = nn.Sequential(*layers)
        self._setup_head(previous, n_classes, n_side, side_hidden)

    @property
    def last_conv(self) -> nn.Conv1d:
        return [m for m in self.features if isinstance(m, nn.Conv1d)][-1]

    def waveform_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x[:, None, :]).mean(dim=-1)


class BasicBlock1D(nn.Module):
    """Two convolutions with batch norm and a shortcut (1x1 convolution when the shape changes)."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, padding=kernel // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity() if stride == 1 and in_channels == out_channels
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet1D(_Classifier):
    def __init__(self, n_classes: int = 4, n_side: int = 0, channels: tuple[int, ...] = (32, 64, 128, 128), kernel: int = 7, side_hidden: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, channels[0], kernel, padding=kernel // 2, bias=False), nn.BatchNorm1d(channels[0]), nn.ReLU())
        stages, previous = [], channels[0]
        for i, out_channels in enumerate(channels):
            stages.append(BasicBlock1D(previous, out_channels, kernel, stride=1 if i == 0 else 2))
            previous = out_channels
        self.stages = nn.Sequential(*stages)
        self._setup_head(previous, n_classes, n_side, side_hidden)

    @property
    def last_conv(self) -> nn.Conv1d:
        return self.stages[-1].conv2

    def waveform_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x[:, None, :])).mean(dim=-1)


def _last_and_mean(output: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Final forward and backward states of the top layer, concatenated with the mean output over time."""
    return torch.cat([hidden[-2], hidden[-1], output.mean(dim=1)], dim=1)


class BiLSTMClassifier(_Classifier):
    def __init__(self, n_classes: int = 4, n_side: int = 0, hidden: int = 64, layers: int = 2, side_hidden: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self._setup_head(4 * hidden, n_classes, n_side, side_hidden)

    def waveform_features(self, x: torch.Tensor) -> torch.Tensor:
        output, (hidden, _) = self.lstm(x[:, :, None])
        return _last_and_mean(output, hidden)


class CNNBiLSTM(_Classifier):
    def __init__(self, n_classes: int = 4, n_side: int = 0, channels: tuple[int, ...] = (32, 64), kernel: int = 7,
                 hidden: int = 64, side_hidden: int = 16) -> None:  # fmt: skip
        super().__init__()
        layers: list[nn.Module] = []
        previous = 1
        for out_channels in channels:
            layers += conv_block(previous, out_channels, kernel)
            previous = out_channels
        self.features = nn.Sequential(*layers)
        self.lstm = nn.LSTM(previous, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self._setup_head(4 * hidden, n_classes, n_side, side_hidden)

    @property
    def last_conv(self) -> nn.Conv1d:
        return [m for m in self.features if isinstance(m, nn.Conv1d)][-1]

    def waveform_features(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.features(x[:, None, :]).transpose(1, 2)
        output, (hidden, _) = self.lstm(sequence)
        return _last_and_mean(output, hidden)

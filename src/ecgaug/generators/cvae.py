"""Conditional VAE (phase3_cvae, PLAN Phase 3).

Variants are settings of one class:

- A ``architecture: conv``: convolutional encoder and decoder; the class one-hot is concatenated to
  the encoder input (as constant channels) and to the latent code; latent 16; the KL weight rises
  linearly from 0 to ``beta`` over the first ``kl_anneal_fraction`` of the steps.
- B as A with a smaller ``beta`` (0.1 and 0.25, both tried; see the runner).
- C as A plus a spectral reconstruction term: L1 distance between the FFT magnitudes of the beat and
  its reconstruction (orthonormal FFT, so it is on the scale of the time-domain error).
- D ``architecture: residual_film``: residual encoder and decoder; the class enters only through
  FiLM (a per-channel scale and shift learned for each class) in every residual block.

Loss per beat: 0.5 * squared error summed over samples (a Gaussian likelihood with unit variance on
z-scored beats) + KL weight * KL divergence [+ spectral weight * spectral L1].
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ecgaug.data.records import CLASSES
from ecgaug.generators.base import BeatGenerator
from ecgaug.generators.layers import SLOPE, UpsampleDecoder, down_stack, flat_size

LOGVAR_RANGE = (-12.0, 8.0)  # clamp for numerical stability


class ConvEncoder(nn.Module):
    """Beat plus class one-hot (as constant input channels) to the mean and log-variance of the latent code."""

    def __init__(self, n_samples: int, n_classes: int, latent: int, channels: list[int], kernel: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.features = down_stack(1 + n_classes, channels, kernel)
        flat = flat_size(self.features, 1 + n_classes, n_samples)
        self.mu = nn.Linear(flat, latent)
        self.logvar = nn.Linear(flat, latent)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        onehot = F.one_hot(labels, self.n_classes).float()[:, :, None].expand(-1, -1, x.shape[1])
        h = self.features(torch.cat([x[:, None, :], onehot], dim=1)).flatten(1)
        return self.mu(h), self.logvar(h).clamp(*LOGVAR_RANGE)


class ConvDecoder(nn.Module):
    """Latent code concatenated with the class one-hot, to a beat."""

    def __init__(self, n_samples: int, n_classes: int, latent: int, channels: list[int], kernel: int) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.net = UpsampleDecoder(latent + n_classes, n_samples, list(reversed(channels)), kernel)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, F.one_hot(labels, self.n_classes).float()], dim=1))


class FiLMBlock(nn.Module):
    """Residual block whose two normalised activations get a class-specific scale and shift (FiLM).

    The FiLM table starts at zero, so every class starts with scale 1 and shift 0.
    """

    def __init__(self, channels: int, n_classes: int, kernel: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels, affine=False)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.norm2 = nn.GroupNorm(8, channels, affine=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.film = nn.Embedding(n_classes, 4 * channels)
        nn.init.zeros_(self.film.weight)

    def forward(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        g1, b1, g2, b2 = self.film(labels)[:, :, None].chunk(4, dim=1)
        out = self.conv1(F.leaky_relu(self.norm1(h) * (1 + g1) + b1, SLOPE))
        out = self.conv2(F.leaky_relu(self.norm2(out) * (1 + g2) + b2, SLOPE))
        return h + out


class FiLMEncoder(nn.Module):
    def __init__(self, n_samples: int, n_classes: int, latent: int, channels: list[int], kernel: int) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, channels[0], kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        previous = channels[0]
        for out_channels in channels:
            self.blocks.append(FiLMBlock(previous, n_classes, kernel))
            self.downs.append(nn.Conv1d(previous, out_channels, kernel, stride=2, padding=kernel // 2))
            previous = out_channels
        with torch.no_grad():
            flat = self._features(torch.zeros(1, n_samples), torch.zeros(1, dtype=torch.long)).shape[1]
        self.mu = nn.Linear(flat, latent)
        self.logvar = nn.Linear(flat, latent)

    def _features(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = self.stem(x[:, None, :])
        for block, down in zip(self.blocks, self.downs):
            h = F.leaky_relu(down(block(h, labels)), SLOPE)
        return h.flatten(1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._features(x, labels)
        return self.mu(h), self.logvar(h).clamp(*LOGVAR_RANGE)


class FiLMDecoder(nn.Module):
    def __init__(self, n_samples: int, n_classes: int, latent: int, channels: list[int], kernel: int) -> None:
        super().__init__()
        channels = list(reversed(channels))
        self.n_samples = n_samples
        self.base = -(-n_samples // 2 ** len(channels))
        self.fc = nn.Linear(latent, channels[0] * self.base)
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        previous = channels[0]
        for out_channels in channels:
            self.ups.append(nn.Conv1d(previous, out_channels, kernel, padding=kernel // 2))
            self.blocks.append(FiLMBlock(out_channels, n_classes, kernel))
            previous = out_channels
        self.out = nn.Conv1d(previous, 1, kernel, padding=kernel // 2)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.fc(z), SLOPE).view(z.shape[0], -1, self.base)
        for up, block in zip(self.ups, self.blocks):
            h = block(F.leaky_relu(up(F.interpolate(h, scale_factor=2, mode="nearest")), SLOPE), labels)
        return self.out(h)[:, 0, : self.n_samples]


ARCHITECTURES = {"conv": (ConvEncoder, ConvDecoder), "residual_film": (FiLMEncoder, FiLMDecoder)}


class CVAE(BeatGenerator):
    """Conditional VAE over all learned classes; samples by decoding standard-normal codes."""

    family = "cvae"

    def __init__(self, settings: dict, n_samples: int) -> None:
        super().__init__(settings, n_samples)
        if settings["architecture"] not in ARCHITECTURES:
            raise ValueError(f"Unknown cVAE architecture {settings['architecture']!r}; choose from {sorted(ARCHITECTURES)}")
        encoder, decoder = ARCHITECTURES[settings["architecture"]]
        self.latent = int(settings["latent_dim"])
        self.beta = float(settings["beta"])
        self.anneal_fraction = float(settings["kl_anneal_fraction"])
        self.spectral_weight = float(settings.get("spectral_weight", 0.0))
        args = (n_samples, len(CLASSES), self.latent, list(settings["channels"]), int(settings["kernel"]))
        self.encoder = encoder(*args)
        self.decoder = decoder(*args)

    def modules(self) -> dict[str, nn.Module]:
        return {"encoder": self.encoder, "decoder": self.decoder}

    def setup_optimizers(self) -> None:
        params = [*self.encoder.parameters(), *self.decoder.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=float(self.settings["lr"]))

    def kl_weight(self, step: int, total: int) -> float:
        """Linear warm-up from 0 to ``beta`` over the first ``kl_anneal_fraction`` of the steps."""
        ramp = self.anneal_fraction * total
        return self.beta * (min(1.0, step / ramp) if ramp > 0 else 1.0)

    def train_step(self, draw: Callable[[], tuple[torch.Tensor, torch.Tensor]], step: int, total: int) -> dict[str, float]:
        x, labels = draw()
        mu, logvar = self.encoder(x, labels)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        x_hat = self.decoder(z, labels)
        reconstruction = 0.5 * ((x - x_hat) ** 2).sum(dim=1).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
        weight = self.kl_weight(step, total)
        loss = reconstruction + weight * kl
        out = {"reconstruction": reconstruction.item(), "kl": kl.item(), "kl_weight": weight}
        if self.spectral_weight:
            spectral = (torch.fft.rfft(x, norm="ortho").abs() - torch.fft.rfft(x_hat, norm="ortho").abs()).abs().sum(dim=1).mean()
            loss = loss + self.spectral_weight * spectral
            out["spectral"] = spectral.item()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        out["loss"] = loss.item()
        return out

    def generate(self, labels: torch.Tensor, noise: torch.Generator) -> torch.Tensor:
        z = torch.randn(labels.shape[0], self.latent, generator=noise).to(self.device)
        return self.decoder(z, labels.to(self.device))

"""Class-conditional denoising diffusion model of beats (DDPM training, DDIM sampling; post hoc, PREREGISTRATION v14).

The current standard for synthetic ECG (e.g. SSSD-ECG; Alcaraz and Strodthoff 2023) is a diffusion model: a network learns to
remove Gaussian noise added to a real beat at a random noise level, and new beats are made by starting from pure noise and
denoising step by step. This module keeps the shared generator interface (``BeatGenerator``): class-balanced batches, the
step budget, checkpoints and divergence checks come from ``base.py``.

- **Forward process:** ``x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps`` with the cosine schedule of Nichol and Dhariwal
  (2021) over ``timesteps`` levels.
- **Denoiser:** a small 1D U-Net predicting ``eps``: residual blocks (GroupNorm, SiLU, convolution) whose features are
  scaled and shifted by an embedding of the noise level plus an embedding of the class (FiLM), two downsamplings by 2
  and skip connections.
- **Classifier-free guidance** (Ho and Salimans 2022): during training the class is replaced by a null class with
  probability ``p_uncond``; at sampling ``eps = eps_null + guidance * (eps_class - eps_null)``.
- **Weights:** an exponential moving average of the denoiser (decay ``ema_decay``) is kept and used for sampling.
- **Sampling:** deterministic DDIM (Song et al. 2021) with ``sample_steps`` evenly spaced levels; the seed fixes the
  starting noise, so the same seed gives the same beats.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import torch
from torch import nn

from ecgaug.data.records import CLASSES
from ecgaug.generators.base import BeatGenerator


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """``abar_t`` for t = 1..T (index 0..T-1), decreasing from about 1 to about 0 (Nichol and Dhariwal 2021, eq. 17)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer noise levels ``t`` [B] to ``[B, dim]``."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    """Two convolutions with a FiLM scale and shift from the conditioning vector; a 1x1 skip when channels change."""

    def __init__(self, c_in: int, c_out: int, cond_dim: int, kernel: int, groups: int) -> None:
        super().__init__()
        pad = kernel // 2
        self.norm1, self.conv1 = nn.GroupNorm(min(groups, c_in), c_in), nn.Conv1d(c_in, c_out, kernel, padding=pad)
        self.norm2, self.conv2 = nn.GroupNorm(min(groups, c_out), c_out), nn.Conv1d(c_out, c_out, kernel, padding=pad)
        self.film = nn.Linear(cond_dim, 2 * c_out)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.act(self.norm1(h)))
        scale, shift = self.film(cond).chunk(2, dim=1)
        out = self.norm2(out) * (1 + scale[:, :, None]) + shift[:, :, None]
        out = self.conv2(self.act(out))
        return out + self.skip(h)


class UNet1D(nn.Module):
    """``eps`` from ``(x_t, t, label)``: down path, middle, up path with skips. ``n_samples`` must divide by 2**len(channels)-1."""

    def __init__(self, n_samples: int, n_classes: int, channels: list[int], kernel: int, embed_dim: int, groups: int) -> None:
        super().__init__()
        levels = len(channels) - 1
        if n_samples % (2**levels):
            raise ValueError(f"{n_samples} samples cannot be halved {levels} times")
        self.embed_dim = embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.SiLU(), nn.Linear(embed_dim * 2, embed_dim))
        self.class_embed = nn.Embedding(n_classes + 1, embed_dim)  # the last index is the null class
        self.stem = nn.Conv1d(1, channels[0], kernel, padding=kernel // 2)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        for i, c in enumerate(channels):
            c_prev = channels[i - 1] if i else channels[0]
            self.down.append(ResBlock(c_prev, c, embed_dim, kernel, groups))
            self.pool.append(nn.Conv1d(c, c, 4, stride=2, padding=1) if i < levels else nn.Identity())
        self.mid = ResBlock(channels[-1], channels[-1], embed_dim, kernel, groups)
        self.up = nn.ModuleList()
        self.unpool = nn.ModuleList()
        for i in reversed(range(levels)):
            self.unpool.append(nn.ConvTranspose1d(channels[i + 1], channels[i], 4, stride=2, padding=1))
            self.up.append(ResBlock(2 * channels[i], channels[i], embed_dim, kernel, groups))
        self.out = nn.Sequential(nn.GroupNorm(min(groups, channels[0]), channels[0]), nn.SiLU(), nn.Conv1d(channels[0], 1, kernel, padding=kernel // 2))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(time_embedding(t, self.embed_dim)) + self.class_embed(labels)
        h = self.stem(x[:, None, :])
        skips = []
        for block, pool in zip(self.down, self.pool):
            h = block(h, cond)
            if not isinstance(pool, nn.Identity):
                skips.append(h)
                h = pool(h)
        h = self.mid(h, cond)
        for unpool, block in zip(self.unpool, self.up):
            h = unpool(h)
            h = block(torch.cat([h, skips.pop()], dim=1), cond)
        return self.out(h)[:, 0, :]


class DiffusionGenerator(BeatGenerator):
    """Class-conditional DDPM over all learned classes, sampled with DDIM and classifier-free guidance."""

    family = "diffusion"

    def __init__(self, settings: dict, n_samples: int) -> None:
        super().__init__(settings, n_samples)
        self.timesteps = int(settings["timesteps"])
        self.sample_steps = int(settings["sample_steps"])
        self.guidance = float(settings["guidance"])
        self.p_uncond = float(settings["p_uncond"])
        self.ema_decay = float(settings["ema_decay"])
        self.clip_x0 = float(settings.get("clip_x0", 10.0))  # a guard on the clean-beat estimate at the noisiest levels
        self.null_class = len(CLASSES)
        self.denoiser = UNet1D(n_samples, len(CLASSES), list(settings["channels"]), int(settings["kernel"]), int(settings["embed_dim"]),
                               int(settings.get("groups", 8)))  # fmt: skip
        self.ema = copy.deepcopy(self.denoiser)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.alpha_bar = cosine_alpha_bar(self.timesteps)

    def modules(self) -> dict[str, nn.Module]:
        return {"denoiser": self.denoiser, "ema": self.ema}

    def setup_optimizers(self) -> None:
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=float(self.settings["lr"]), weight_decay=float(self.settings.get("weight_decay", 0.0)))

    def _abar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(t.device)[t]

    def train_step(self, draw: Callable[[], tuple[torch.Tensor, torch.Tensor]], step: int, total: int) -> dict[str, float]:
        x0, labels = draw()
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        abar = self._abar(t)[:, None]
        x_t = abar.sqrt() * x0 + (1 - abar).sqrt() * eps
        drop = torch.rand(labels.shape[0], device=labels.device) < self.p_uncond
        cond = torch.where(drop, torch.full_like(labels, self.null_class), labels)
        loss = ((self.denoiser(x_t, t, cond) - eps) ** 2).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.optimizer.step()
        with torch.no_grad():
            for p_ema, p in zip(self.ema.parameters(), self.denoiser.parameters()):
                p_ema.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)
            for b_ema, b in zip(self.ema.buffers(), self.denoiser.buffers()):
                b_ema.copy_(b)
        return {"loss": loss.item()}

    def ddim_levels(self) -> list[int]:
        """The noise levels visited by DDIM, from the noisiest to the cleanest (0-based indices into ``alpha_bar``)."""
        levels = torch.linspace(self.timesteps - 1, 0, self.sample_steps).round().long().tolist()
        return list(dict.fromkeys(levels))

    @torch.no_grad()
    def generate(self, labels: torch.Tensor, noise: torch.Generator) -> torch.Tensor:
        n = labels.shape[0]
        x = torch.randn(n, self.n_samples, generator=noise).to(self.device)
        labels = labels.to(self.device)
        null = torch.full_like(labels, self.null_class)
        net = self.ema
        levels = self.ddim_levels()
        for i, level in enumerate(levels):
            t = torch.full((n,), level, device=self.device, dtype=torch.long)
            if self.guidance == 1.0:
                eps = net(x, t, labels)
            else:
                both = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([labels, null]))
                eps_c, eps_u = both[:n], both[n:]
                eps = eps_u + self.guidance * (eps_c - eps_u)
            abar = self._abar(t)[:, None]
            x0 = ((x - (1 - abar).sqrt() * eps) / abar.sqrt()).clamp(-self.clip_x0, self.clip_x0)
            eps = (x - abar.sqrt() * x0) / (1 - abar).sqrt()  # consistent with the clipped estimate
            abar_next = self._abar(torch.full((n,), levels[i + 1], device=self.device, dtype=torch.long))[:, None] if i + 1 < len(levels) else torch.ones_like(abar)
            x = abar_next.sqrt() * x0 + (1 - abar_next).sqrt() * eps
        return x

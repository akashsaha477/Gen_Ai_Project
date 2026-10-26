"""Conditional WGAN-GP (phase3_wgan_gp, PLAN Phase 3).

Shared settings (Gulrajani et al., 2017): 5 critic updates per generator update, gradient-penalty
weight 10, Adam with learning rate 1e-4 and betas (0, 0.9), no batch norm in the critic. A "step" is
one generator update.

Variants are settings of one class (``conditioning``):

- A ``concat``: a label embedding is concatenated to the generator noise, and a learned per-class
  sequence is added to the critic input as a second channel.
- B ``projection``: projection critic; its score adds the inner product of a class embedding with the
  critic features (Miyato and Koyama, 2018). The generator is conditioned as in A.
- C ``auxiliary``: AC-WGAN-GP; the critic has an auxiliary class head trained on real beats, and the
  generator is also rewarded when its beats are classified as the intended class.
- D ``per_class``: a separate unconditional WGAN-GP for each of S, V and F. The step budget is shared:
  step t trains the GAN of class t mod 3, so each class gets a third of the steps and every batch is
  of that class only.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ecgaug.data.records import CLASS_INDEX, CLASSES
from ecgaug.generators.base import GENERATED_CLASSES, BeatGenerator
from ecgaug.generators.layers import UpsampleDecoder, down_stack, flat_size

CONDITIONING = ("concat", "projection", "auxiliary", "per_class")


class Generator(nn.Module):
    """Noise (plus a label embedding if conditional) to a beat."""

    def __init__(self, n_samples: int, noise_dim: int, n_classes: int, embed_dim: int, channels: list[int], kernel: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim) if n_classes else None
        self.net = UpsampleDecoder(noise_dim + (embed_dim if n_classes else 0), n_samples, channels, kernel)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        h = torch.cat([z, self.embed(labels)], dim=1) if self.embed is not None else z
        return self.net(h)


class Critic(nn.Module):
    """Convolutional critic without normalisation. Returns ``(score, class logits or None)``."""

    def __init__(self, n_samples: int, n_classes: int, conditioning: str, channels: list[int], kernel: int) -> None:
        super().__init__()
        self.conditioning = conditioning
        in_channels = 2 if conditioning == "concat" else 1
        self.label_channel = nn.Embedding(n_classes, n_samples) if conditioning == "concat" else None
        self.features = down_stack(in_channels, channels, kernel)
        flat = flat_size(self.features, in_channels, n_samples)
        self.score = nn.Linear(flat, 1)
        self.projection = nn.Embedding(n_classes, flat) if conditioning == "projection" else None
        if self.projection is not None:
            nn.init.normal_(self.projection.weight, std=0.02)
        self.classifier = nn.Linear(flat, n_classes) if conditioning == "auxiliary" else None

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = x[:, None, :]
        if self.label_channel is not None:
            h = torch.cat([h, self.label_channel(labels)[:, None, :]], dim=1)
        f = self.features(h).flatten(1)
        score = self.score(f)[:, 0]
        if self.projection is not None:
            score = score + (self.projection(labels) * f).sum(dim=1)
        return score, (self.classifier(f) if self.classifier is not None else None)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    """Mean of (||grad of the critic at random interpolates|| - 1)^2 (Gulrajani et al., 2017)."""
    alpha = torch.rand(real.shape[0], 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score, _ = critic(mixed, labels)
    (grad,) = torch.autograd.grad(score.sum(), mixed, create_graph=True)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


class WGANGP(BeatGenerator):
    family = "wgan_gp"

    def __init__(self, settings: dict, n_samples: int) -> None:
        super().__init__(settings, n_samples)
        self.conditioning = settings["conditioning"]
        if self.conditioning not in CONDITIONING:
            raise ValueError(f"Unknown conditioning {self.conditioning!r}; choose from {CONDITIONING}")
        self.noise_dim = int(settings["noise_dim"])
        g_channels, c_channels, kernel = list(settings["generator_channels"]), list(settings["critic_channels"]), int(settings["kernel"])
        if self.conditioning == "per_class":
            self.pairs = {
                c: (Generator(n_samples, self.noise_dim, 0, 0, g_channels, kernel), Critic(n_samples, 0, "none", c_channels, kernel))
                for c in GENERATED_CLASSES
            }
        else:
            n = len(CLASSES)
            self.pairs = {
                "all": (
                    Generator(n_samples, self.noise_dim, n, int(settings["embed_dim"]), g_channels, kernel),
                    Critic(n_samples, n, self.conditioning, c_channels, kernel),
                )
            }

    def modules(self) -> dict[str, nn.Module]:
        out: dict[str, nn.Module] = {}
        for key, (generator, critic) in self.pairs.items():
            out[f"generator_{key}"] = generator
            out[f"critic_{key}"] = critic
        return out

    def setup_optimizers(self) -> None:
        lr, betas = float(self.settings["lr"]), tuple(self.settings["adam_betas"])
        self.optimizers = {
            key: (torch.optim.Adam(g.parameters(), lr=lr, betas=betas), torch.optim.Adam(c.parameters(), lr=lr, betas=betas))
            for key, (g, c) in self.pairs.items()
        }

    def _key(self, step: int) -> str:
        return GENERATED_CLASSES[(step - 1) % len(GENERATED_CLASSES)] if self.conditioning == "per_class" else "all"

    def batch_classes(self, step: int) -> tuple[int, ...]:
        if self.conditioning == "per_class":
            return (CLASS_INDEX[self._key(step)],)
        return self.class_ids

    def sample_classes(self) -> tuple[str, ...]:
        return GENERATED_CLASSES if self.conditioning == "per_class" else super().sample_classes()

    def _labels(self, labels: torch.Tensor) -> torch.Tensor | None:
        return None if self.conditioning == "per_class" else labels

    def train_step(self, draw: Callable[[], tuple[torch.Tensor, torch.Tensor]], step: int, total: int) -> dict[str, float]:
        key = self._key(step)
        generator, critic = self.pairs[key]
        opt_g, opt_c = self.optimizers[key]
        aux = float(self.settings.get("auxiliary_weight", 1.0)) if self.conditioning == "auxiliary" else 0.0
        for _ in range(int(self.settings["n_critic"])):
            x, labels = draw()
            cond = self._labels(labels)
            z = torch.randn(x.shape[0], self.noise_dim, device=self.device)
            with torch.no_grad():
                fake = generator(z, cond)
            real_score, real_logits = critic(x, cond)
            fake_score, _ = critic(fake, cond)
            penalty = gradient_penalty(critic, x, fake, cond)
            critic_loss = fake_score.mean() - real_score.mean() + float(self.settings["gradient_penalty"]) * penalty
            if aux:
                critic_loss = critic_loss + aux * F.cross_entropy(real_logits, labels)
            opt_c.zero_grad(set_to_none=True)
            critic_loss.backward()
            opt_c.step()
        _, labels = draw()
        cond = self._labels(labels)
        z = torch.randn(labels.shape[0], self.noise_dim, device=self.device)
        score, logits = critic(generator(z, cond), cond)
        generator_loss = -score.mean()
        if aux:
            generator_loss = generator_loss + aux * F.cross_entropy(logits, labels)
        opt_g.zero_grad(set_to_none=True)
        generator_loss.backward()
        opt_g.step()
        suffix = f"_{key}" if self.conditioning == "per_class" else ""
        return {
            f"critic_loss{suffix}": critic_loss.item(),
            f"wasserstein{suffix}": (real_score.mean() - fake_score.mean()).item(),
            f"gradient_penalty{suffix}": penalty.item(),
            f"generator_loss{suffix}": generator_loss.item(),
        }

    def generate(self, labels: torch.Tensor, noise: torch.Generator) -> torch.Tensor:
        z = torch.randn(labels.shape[0], self.noise_dim, generator=noise).to(self.device)
        labels = labels.to(self.device)
        if self.conditioning != "per_class":
            return self.pairs["all"][0](z, labels)
        out = torch.empty(labels.shape[0], self.n_samples, device=self.device)
        for index in labels.unique().tolist():
            mask = labels == index
            out[mask] = self.pairs[CLASSES[index]][0](z[mask], None)
        return out

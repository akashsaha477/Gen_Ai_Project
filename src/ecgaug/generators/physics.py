"""McSharry et al.'s (2003) ECG simulator as a prior for generated beats (PREREGISTRATION §10 v20, P21; post hoc).

The simulator moves a point around the unit circle (angle theta, R peak at theta = 0) while the ECG value z follows

    dz/dt = -sum_i a_i dtheta_i exp(-dtheta_i^2 / (2 b_i^2)) - (z - z0),    i in {P, Q, R, S, T},

with dtheta_i = theta - theta_i wrapped to (-pi, pi]. On the circle theta(t) = omega t, so the x and y equations need no
integration. SimGANs (Golany, Radinsky and Freedman, ICML 2020) integrate it with explicit Euler steps and score a
generated beat by how far it is from satisfying the z equation at every step (the Euler loss). Here the beat window is
the frozen representation's (R peak at the centre sample, 360 Hz) and omega is 2 pi over the beat's local RR interval.

A parameter vector ``eta`` holds, in order, the five positions theta_i, the five heights a_i, the five widths b_i and the
baseline z0 (16 numbers).
"""

from __future__ import annotations

import math

import numpy as np
import torch

WAVES: tuple[str, ...] = ("P", "Q", "R", "S", "T")
N_ETA = 3 * len(WAVES) + 1
FS = 360.0
DT = 1.0 / FS
MIN_WIDTH = 0.01  # radians; a narrower bump is a spike, not a wave
# McSharry et al. (2003), table 1: the normal beat's positions (radians), heights and widths.
MCSHARRY_POSITIONS = (-math.pi / 3, -math.pi / 12, 0.0, math.pi / 12, math.pi / 2)
MCSHARRY_HEIGHTS = (1.2, -5.0, 30.0, -7.5, 0.75)
MCSHARRY_WIDTHS = (0.25, 0.1, 0.1, 0.1, 0.4)


def split_eta(eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(positions [B, 5], heights [B, 5], widths [B, 5], baseline [B])`` of ``eta`` [B, 16]."""
    k = len(WAVES)
    return eta[:, :k], eta[:, k : 2 * k], eta[:, 2 * k : 3 * k], eta[:, 3 * k]


def join_eta(positions: torch.Tensor, heights: torch.Tensor, widths: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    return torch.cat([positions, heights, widths, baseline.reshape(-1, 1)], dim=1)


def phase(omega: torch.Tensor, n_samples: int) -> torch.Tensor:
    """theta [B, L] of every sample: omega times the time from the R peak (the centre sample, index ``n_samples // 2``)."""
    t = (torch.arange(n_samples, dtype=omega.dtype, device=omega.device) - n_samples // 2) * DT
    return omega.reshape(-1, 1) * t


def wrap(angle: torch.Tensor) -> torch.Tensor:
    """Angles in [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def bumps(theta: torch.Tensor, positions: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """dtheta_i exp(-dtheta_i^2 / (2 b_i^2)) [B, 5, L]: each wave's term of the z equation with unit height."""
    d = wrap(theta[:, None, :] - positions[:, :, None])
    return d * torch.exp(-(d**2) / (2 * widths[:, :, None] ** 2))


def forcing(theta: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """sum_i a_i dtheta_i exp(...) [B, L] (the z equation is dz/dt = -forcing - (z - z0))."""
    positions, heights, widths, _ = split_eta(eta)
    return (heights[:, :, None] * bumps(theta, positions, widths.clamp_min(MIN_WIDTH))).sum(dim=1)


def euler_matrices(n_samples: int, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """``(decay [L], M [L, L])`` with explicit Euler steps z_{l+1} = (1 - dt) z_l + dt u_l written out:
    z = decay * z_0 + M @ u, decay_l = (1 - dt)^l and M[l, k] = dt (1 - dt)^(l - 1 - k) for k < l."""
    q = 1.0 - DT
    idx = torch.arange(n_samples, dtype=dtype, device=device)
    power = idx[:, None] - 1 - idx[None, :]
    m = torch.where(power >= 0, DT * q ** power.clamp_min(0), torch.zeros((), dtype=dtype, device=device))
    return q**idx, m


def simulate(eta: torch.Tensor, omega: torch.Tensor, z_start: torch.Tensor, n_samples: int) -> torch.Tensor:
    """The simulator's beat [B, L] from ``z_start`` by explicit Euler steps of 1/360 s."""
    decay, m = euler_matrices(n_samples, eta.dtype, eta.device)
    u = split_eta(eta)[3].reshape(-1, 1) - forcing(phase(omega, n_samples), eta)
    return decay * z_start.reshape(-1, 1) + u @ m.T


def euler_distance(beats: torch.Tensor, eta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """SimGANs' simulator distance [B]: sum over steps of ((h_{l+1} - h_l) / dt - f_z(h_l))^2, f_z the z equation's
    right-hand side. It is 0 for the simulator's own beat and linear in the beat inside the square."""
    theta = phase(omega.to(beats.dtype), beats.shape[1])
    f = forcing(theta, eta.to(beats.dtype))
    z0 = split_eta(eta.to(beats.dtype))[3].reshape(-1, 1)
    residual = (beats[:, 1:] - beats[:, :-1]) / DT + f[:, :-1] + beats[:, :-1] - z0
    return (residual**2).sum(dim=1)


def _design(theta: torch.Tensor, positions: torch.Tensor, widths: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Columns [B, L, 6] of the simulated beat's linear part: -M @ bump_i for the five heights, M @ 1 for the baseline."""
    waves = -(bumps(theta, positions, widths) @ m.T)  # [B, 5, L]
    base = (m @ torch.ones(m.shape[0], dtype=m.dtype, device=m.device)).expand(theta.shape[0], -1)
    return torch.cat([waves, base[:, None, :]], dim=1).transpose(1, 2)


def _solve(design: torch.Tensor, target: torch.Tensor, ridge: float = 1e-9) -> torch.Tensor:
    """Least-squares coefficients [B, 6], by the normal equations with a tiny ridge (differentiable)."""
    gram = design.transpose(1, 2) @ design
    scale = gram.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
    eye = torch.eye(gram.shape[1], dtype=gram.dtype, device=gram.device)
    rhs = design.transpose(1, 2) @ target[:, :, None]
    return torch.linalg.solve(gram + ridge * scale[:, None, None] * eye, rhs)[:, :, 0]


def fit(beats: np.ndarray, omega: np.ndarray, iterations: int = 500, lr: float = 0.01, chunk: int = 2048) -> dict[str, np.ndarray]:
    """Per beat, the simulator parameters whose beat (from the beat's first sample) is closest in least squares.

    Heights and baseline are solved exactly for the current positions and widths, which Adam fits from McSharry et al.'s
    normal-beat values. Returns ``eta`` [n, 16], the simulated beats ``fitted`` [n, L] and ``r2`` [n] (float64)."""
    x_all = torch.as_tensor(np.asarray(beats, dtype=np.float64))
    w_all = torch.as_tensor(np.asarray(omega, dtype=np.float64))
    n, length = x_all.shape
    decay, m = euler_matrices(length)
    etas, fits = [], []
    for start in range(0, n, chunk):
        x, w = x_all[start : start + chunk], w_all[start : start + chunk]
        b = x.shape[0]
        positions = torch.tensor(MCSHARRY_POSITIONS, dtype=torch.float64).repeat(b, 1).requires_grad_(True)
        log_width = torch.log(torch.tensor(MCSHARRY_WIDTHS, dtype=torch.float64) - MIN_WIDTH).repeat(b, 1).requires_grad_(True)
        optimizer = torch.optim.Adam([positions, log_width], lr=lr)
        target = x - decay * x[:, :1]
        theta = phase(w, length)
        for step in range(iterations + 1):
            widths = MIN_WIDTH + torch.exp(log_width)
            design = _design(theta, positions, widths, m)
            coef = _solve(design, target)
            pred = decay * x[:, :1] + (design @ coef[:, :, None])[:, :, 0]
            if step == iterations:
                break
            loss = ((x - pred) ** 2).sum(dim=1).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            widths = MIN_WIDTH + torch.exp(log_width)
            etas.append(join_eta(positions.detach(), coef[:, :5].detach(), widths, coef[:, 5].detach()))
            fits.append(pred.detach())
    eta = torch.cat(etas).numpy()
    fitted = torch.cat(fits).numpy()
    x_np = x_all.numpy()
    sst = ((x_np - x_np.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2 = 1.0 - ((x_np - fitted) ** 2).sum(axis=1) / np.where(sst > 0, sst, np.nan)
    return {"eta": eta, "fitted": fitted, "r2": r2}


def omega_of(rr_prev: np.ndarray, rr_ratio: np.ndarray) -> np.ndarray:
    """2 pi over the local RR interval in seconds (``rr_prev`` over v2's ratio: the median RR interval near the beat)."""
    local = np.asarray(rr_prev, dtype=np.float64) / np.asarray(rr_ratio, dtype=np.float64)
    return 2 * math.pi / local


def class_distribution(eta: np.ndarray) -> dict[str, list[float]]:
    """SimGANs' class distribution: a Gaussian with diagonal covariance over the fitted parameter vectors."""
    e = np.asarray(eta, dtype=np.float64)
    return {"mean": e.mean(axis=0).tolist(), "sd": e.std(axis=0).tolist()}


def draw_eta(mean: torch.Tensor, sd: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """``n`` parameter vectors from the class Gaussian (widths kept at least ``MIN_WIDTH`` by ``forcing``)."""
    noise = torch.randn(n, mean.shape[0], generator=generator, dtype=mean.dtype, device="cpu" if generator is not None else mean.device)
    return mean + sd * noise.to(mean.device)


def kappa(beats: np.ndarray, omega: np.ndarray, distribution: dict[str, list[float]], seed: int = 0) -> float:
    """The mean Euler distance of real beats with parameters drawn from their class distribution (one draw per beat)."""
    x = torch.as_tensor(np.asarray(beats, dtype=np.float64))
    mean = torch.tensor(distribution["mean"], dtype=torch.float64)
    sd = torch.tensor(distribution["sd"], dtype=torch.float64)
    eta = draw_eta(mean, sd, x.shape[0], torch.Generator().manual_seed(int(seed)))
    return float(euler_distance(x, eta, torch.as_tensor(np.asarray(omega, dtype=np.float64))).mean())


class EulerPrior:
    """The Euler loss of P21 for generated patient-relative beats (PREREGISTRATION §10 v20).

    A generated beat of class c (in the generator's scaled units) is multiplied back by ``scale`` and gets the template
    of a training beat of class c drawn at random; that beat's omega and a parameter vector from the class distribution
    give the Euler distance, divided by the class's ``kappa`` (real beats score 1 on average). Classes without a pool
    (N) get no loss. Random draws come from torch's global generator, which ``BeatGenerator.fit`` seeds."""

    def __init__(self, classes: dict[int, dict], scale: float) -> None:
        self.scale = float(scale)
        self.classes = classes  # class index -> {"templates", "omega", "mean", "sd", "kappa"} (tensors except kappa)

    def to(self, device: torch.device | str) -> EulerPrior:
        self.classes = {c: {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in entry.items()} for c, entry in self.classes.items()}
        return self

    @classmethod
    def build(cls, pools: dict[int, tuple[np.ndarray, np.ndarray]], physics: dict[int, dict], scale: float) -> EulerPrior:
        """``pools[c] = (templates [n, L], omega [n])``; ``physics[c] = {"mean", "sd", "kappa"}``."""
        classes = {}
        for c, (templates, omega) in pools.items():
            p = physics[c]
            classes[int(c)] = {"templates": torch.as_tensor(np.asarray(templates, dtype=np.float32)),
                               "omega": torch.as_tensor(np.asarray(omega, dtype=np.float32)),
                               "mean": torch.tensor(p["mean"], dtype=torch.float32), "sd": torch.tensor(p["sd"], dtype=torch.float32),
                               "kappa": float(p["kappa"])}  # fmt: skip
        return cls(classes, scale)

    def __call__(self, beats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
        terms = []
        for c, entry in self.classes.items():
            mask = labels == c
            n = int(mask.sum())
            if n == 0:
                continue
            pick = torch.randint(entry["templates"].shape[0], (n,), device=beats.device)
            eta = entry["mean"] + entry["sd"] * torch.randn(n, entry["mean"].shape[0], device=beats.device)
            whole = beats[mask] * self.scale + entry["templates"][pick]
            terms.append(euler_distance(whole, eta, entry["omega"][pick]) / entry["kappa"])
        return torch.cat(terms).mean() if terms else None

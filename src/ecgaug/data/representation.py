"""Input representations: one config-driven code path for every phase1_input_representation variant.

A representation turns the continuous MLII signal of a record plus R-peak indices into one fixed-length
window per beat. Only the representation differs between phase1_input_representation variants; the beat set, labels and
splits do not, because the usable-beat rule below never depends on the representation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as sps

from ecgaug.data.records import FS

# A beat is usable if the widest candidate window (variant B) fits inside the record and both RR
# intervals exist. Applied identically to every representation.
USABLE_PRE_S = 0.4
USABLE_POST_S = 0.6


@dataclass(frozen=True)
class RepresentationSpec:
    """Definition of an input representation (stored verbatim in ``contract.json``)."""

    name: str
    window_pre_s: float
    window_post_s: float
    bandpass_hz: tuple[float, float] | None = None
    bandpass_order: int = 4
    target_fs: int | None = None
    normalization: str = "per_beat_zscore"
    rr_features: bool = False

    def __post_init__(self) -> None:
        if self.bandpass_hz is not None:
            object.__setattr__(self, "bandpass_hz", tuple(float(v) for v in self.bandpass_hz))
        if self.normalization != "per_beat_zscore":
            raise ValueError(f"Unknown normalization {self.normalization!r}")
        if self.window_pre_s > USABLE_PRE_S or self.window_post_s > USABLE_POST_S:
            raise ValueError("Window exceeds the usable-beat margins; widen USABLE_PRE_S/USABLE_POST_S first")

    @property
    def effective_fs(self) -> int:
        return int(self.target_fs or FS)

    @property
    def window_samples(self) -> tuple[int, int]:
        """Samples before and after the R peak at the effective sampling rate."""
        fs = self.effective_fs
        total = math.ceil((self.window_pre_s + self.window_post_s) * fs - 1e-9)
        pre = math.floor(self.window_pre_s * fs + 1e-9)
        return pre, total - pre

    @property
    def n_samples(self) -> int:
        return sum(self.window_samples)

    @property
    def n_inputs(self) -> int:
        """Input values per beat: samples plus RR side features (phase1_input_representation tie-break)."""
        return self.n_samples + (3 if self.rr_features else 0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bandpass_hz"] = list(self.bandpass_hz) if self.bandpass_hz else None
        data.update(effective_fs=self.effective_fs, window_samples=list(self.window_samples), n_samples=self.n_samples)
        return data


def usable_mask(r_idx: np.ndarray, n_signal: int, rr_prev: np.ndarray, rr_next: np.ndarray) -> np.ndarray:
    """Beats whose widest candidate window fits in the record and that have both RR intervals."""
    pre, post = round(USABLE_PRE_S * FS), round(USABLE_POST_S * FS)
    r_idx = np.asarray(r_idx)
    fits = (r_idx - pre >= 0) & (r_idx + post <= n_signal)
    return fits & np.isfinite(rr_prev) & np.isfinite(rr_next)


def bandpass(x: np.ndarray, band_hz: tuple[float, float], order: int, fs: int = FS) -> np.ndarray:
    """Zero-phase Butterworth band-pass of a continuous record."""
    sos = sps.butter(order, band_hz, btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def zscore_per_beat(windows: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Subtract each window's mean and divide by its standard deviation (flat windows become zeros)."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return (windows - mean) / np.maximum(std, eps)


def transform_record(signal: np.ndarray, r_idx: np.ndarray, spec: RepresentationSpec) -> np.ndarray:
    """Windows ``[n_beats, spec.n_samples]`` (float32) for the beats at ``r_idx`` of one record.

    Order of operations: optional band-pass on the continuous record, optional polyphase resampling
    of the continuous record, windowing around R, per-beat z-score. All beats must be usable.
    """
    x = np.asarray(signal, dtype=np.float64)
    r = np.asarray(r_idx, dtype=np.int64)
    if spec.bandpass_hz is not None:
        x = bandpass(x, spec.bandpass_hz, spec.bandpass_order)
    if spec.target_fs is not None and spec.target_fs != FS:
        divisor = math.gcd(spec.target_fs, FS)
        x = sps.resample_poly(x, spec.target_fs // divisor, FS // divisor)
        r = np.rint(r * spec.target_fs / FS).astype(np.int64)
    pre, post = spec.window_samples
    if r.size and (r.min() - pre < 0 or r.max() + post > x.size):
        raise ValueError("A window runs past the record edge; filter beats with usable_mask first")
    windows = x[r[:, None] + np.arange(-pre, post)[None, :]]
    return zscore_per_beat(windows).astype(np.float32)

"""Saliency methods of phase5_explainability (PLAN Phase 5).

| variant | method | applies to |
|---|---|---|
| A | Grad-CAM 1D on the last convolution layer, upsampled to the beat length | models with ``last_conv`` (the cnn) |
| B | Integrated gradients (captum), 50 steps, averaged over a zero baseline and a per-beat-mean baseline | every model |
| C | SmoothGrad (captum NoiseTunnel over gradients), 25 samples, noise SD 0.1 x the beat's SD | every model |
| D | Occlusion (captum), sliding window of 50 ms, stride 25 ms, zero baseline | every model |

Every method explains the class the model predicts for the beat and returns one value per input
sample, ``[n_beats, length]``; the RR side features are passed through but not attributed. Saliency is
computed on the CPU with cuDNN off, because cuDNN's LSTM cannot backpropagate in evaluation mode.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from ecgaug.eval.plotting import import_pyplot

METHODS = ("gradcam", "integrated_gradients", "smoothgrad", "occlusion")


@contextmanager
def _explainable(model: nn.Module):
    """Model on the CPU in evaluation mode, cuDNN off (restored afterwards)."""
    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    model.cpu().eval()
    try:
        yield model
    finally:
        torch.backends.cudnn.enabled = previous


class _Wrapped(nn.Module):
    """``forward(x, side)`` with ``x`` ``[n, L]``, as captum calls it."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, side)


def applies(method: str, model: nn.Module) -> bool:
    return method != "gradcam" or hasattr(model, "last_conv")


def gradcam(model: nn.Module, x: torch.Tensor, side: torch.Tensor | None, target: torch.Tensor) -> np.ndarray:
    """Grad-CAM: ReLU of the gradient-weighted sum of the last convolution's feature maps."""
    stash: dict = {}
    layer = model.last_conv
    forward = layer.register_forward_hook(lambda m, i, o: stash.__setitem__("act", o))
    backward = layer.register_full_backward_hook(lambda m, gi, go: stash.__setitem__("grad", go[0]))
    try:
        x = x.clone().requires_grad_(True)
        logits = model(x, side)
        model.zero_grad()
        logits.gather(1, target[:, None]).sum().backward()
    finally:
        forward.remove()
        backward.remove()
    weights = stash["grad"].mean(dim=2, keepdim=True)
    cam = torch.relu((weights * stash["act"]).sum(dim=1)).detach()
    cam = torch.nn.functional.interpolate(cam[:, None, :], size=x.shape[1], mode="linear", align_corners=False)[:, 0]
    return cam.numpy()


def saliency(method: str, model: nn.Module, x: np.ndarray, side: np.ndarray | None, settings: dict) -> np.ndarray:
    """Saliency ``[n, L]`` of ``method`` for the predicted class of each beat."""
    import_pyplot()  # captum imports matplotlib; this avoids a macOS font-cache failure
    from captum.attr import IntegratedGradients, NoiseTunnel, Occlusion, Saliency

    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if not applies(method, model):
        raise ValueError(f"{method} needs a convolutional model (one with last_conv)")
    out = []
    with _explainable(model):
        wrapped = _Wrapped(model)
        batch = int(settings.get("batch_size", 64))
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(x[start : start + batch], dtype=torch.float32)
            sb = None if side is None else torch.as_tensor(side[start : start + batch], dtype=torch.float32)
            with torch.no_grad():
                target = wrapped(xb, sb).argmax(dim=1)
            extra = (sb,) if sb is not None else None
            if method == "gradcam":
                values = gradcam(model, xb, sb, target)
            elif method == "integrated_gradients":
                ig = IntegratedGradients(wrapped)
                steps = int(settings["ig_steps"])
                zero = ig.attribute(xb, baselines=torch.zeros_like(xb), target=target, additional_forward_args=extra, n_steps=steps)
                mean = ig.attribute(xb, baselines=xb.mean(dim=1, keepdim=True).expand_as(xb), target=target, additional_forward_args=extra, n_steps=steps)
                values = ((zero + mean) / 2).detach().numpy()
            elif method == "smoothgrad":
                tunnel = NoiseTunnel(Saliency(wrapped))
                sd = float(settings["smoothgrad_sd"]) * float(xb.std(dim=1).mean())
                values = tunnel.attribute(xb, nt_type="smoothgrad", nt_samples=int(settings["smoothgrad_samples"]), stdevs=sd,
                                          target=target, additional_forward_args=extra).detach().numpy()  # fmt: skip
            else:
                fs = float(settings["fs"])
                window = max(1, int(round(settings["occlusion_window_ms"] * fs / 1000)))
                stride = max(1, int(round(settings["occlusion_stride_ms"] * fs / 1000)))
                values = Occlusion(wrapped).attribute(xb, sliding_window_shapes=(window,), strides=(stride,), baselines=0.0,
                                                      target=target, additional_forward_args=extra).detach().numpy()  # fmt: skip
            out.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros((0, x.shape[1]), dtype=np.float32)

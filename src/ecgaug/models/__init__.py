"""Classifier architectures and the helpers that build them and transfer their weights."""

from __future__ import annotations

from torch import nn

from ecgaug.models.probe import ProbeCNN, count_parameters

__all__ = ["ProbeCNN", "build_model", "count_parameters", "load_encoder"]

HEAD_PREFIX = "head."


def build_model(model_cfg: dict, n_classes: int, n_side: int = 0) -> nn.Module:
    """Build the architecture named in a config's ``model`` block.

    Every architecture keeps its final classification layer in an attribute called ``head``; that is
    what :func:`load_encoder` leaves out when weights are transferred.
    """
    name = model_cfg.get("name", "probe_cnn")
    if name == "probe_cnn":
        return ProbeCNN(
            n_classes=n_classes,
            n_side=n_side,
            channels=tuple(model_cfg["channels"]),
            kernel=model_cfg["kernel"],
            side_hidden=model_cfg["side_hidden"],
        )
    raise ValueError(f"Unknown model {name!r}")


def load_encoder(model: nn.Module, state_dict: dict) -> list[str]:
    """Copy every tensor of ``state_dict`` except the classification head into ``model``.

    Raises if the two networks differ anywhere outside the head, so a partial transfer cannot pass
    silently. Returns the names of the tensors that were copied.
    """
    own = model.state_dict()
    encoder = {k: v for k, v in state_dict.items() if not k.startswith(HEAD_PREFIX)}
    missing = sorted(k for k in own if not k.startswith(HEAD_PREFIX) and k not in encoder)
    unexpected = sorted(k for k in encoder if k not in own)
    mismatched = sorted(k for k in encoder if k in own and own[k].shape != encoder[k].shape)
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Pretrained weights do not fit the model: missing {missing}, unexpected {unexpected}, "
            f"different shape {mismatched}"
        )
    model.load_state_dict(encoder, strict=False)
    return sorted(encoder)

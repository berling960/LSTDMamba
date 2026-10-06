"""Effective-number class weighting for the hierarchical objective (Eqs. 15-16).

The four levels differ substantially in sample count, so each head uses
effective-number class weights. For class ``k`` at level ``l`` with training
count ``n_{l,k}``::

    q_{l,k} = (1 - beta) / (1 - beta^{n_{l,k}})

All weights are normalized by their mean, and the training objective is the
sum of the four weighted cross-entropies::

    L_task = sum_l CE_{q_l}(O_l, Y_l)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS


def effective_number_weights(
    class_counts: Sequence[int],
    beta: float = 0.9999,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute ``q_{l,k}`` from per-class training counts.

    ``beta`` controls how aggressively rare classes are up-weighted: it is close
    to 1 so that the weight follows the inverse effective number rather than a
    hard inverse-frequency ratio.
    """
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must lie in [0, 1), got {beta}.")
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if counts.ndim != 1:
        raise ValueError("class_counts must be one-dimensional.")
    if torch.any(counts < 0):
        raise ValueError("class_counts must be non-negative.")

    effective = 1.0 - torch.pow(beta, counts.clamp(min=0))
    weights = (1.0 - beta) / effective
    # Classes with no training samples receive a zero weight.
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    if normalize:
        mean = weights.mean()
        if mean > 0:
            weights = weights / mean
    return weights.to(dtype=torch.float32)


def load_class_counts(
    class_counts: Sequence[int] | None = None,
    class_counts_path: str | None = None,
    level_key: str | None = None,
) -> Sequence[int]:
    """Resolve class counts from either explicit values or a JSON file."""
    if class_counts is not None and class_counts_path is not None:
        raise ValueError("Set class_counts or class_counts_path, not both.")
    if class_counts_path is not None:
        path = Path(class_counts_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Class-count file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            if level_key is None:
                raise ValueError(
                    f"{path} is a mapping, so level_key is required."
                )
            if level_key not in data:
                raise KeyError(
                    f"Class-count file {path} has no entry for {level_key!r}."
                )
            data = data[level_key]
        class_counts = data
    if class_counts is None:
        raise ValueError(
            "EffectiveNumberCrossEntropyLoss requires class_counts or "
            "class_counts_path."
        )
    return class_counts


@MODELS.register_module()
class EffectiveNumberCrossEntropyLoss(nn.Module):
    """Cross-entropy weighted by the effective number of samples (Eq. 15).

    ``mode`` selects the weighting scheme:

    * ``"effective"`` -- the ``q_{l,k}`` weights of Eq. (15);
    * ``"none"`` -- an unweighted baseline used for ablations.
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: Sequence[int] | None = None,
        class_counts_path: str | None = None,
        level_key: str | None = None,
        beta: float = 0.9999,
        ignore_index: int = 255,
        mode: str = "effective",
        loss_weight: float = 1.0,
        background_weight: float | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"effective", "none"}:
            raise ValueError(
                f"mode must be 'effective' or 'none', got {mode!r}."
            )
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.mode = mode
        self.loss_weight = float(loss_weight)

        if mode == "effective":
            counts = load_class_counts(
                class_counts=class_counts,
                class_counts_path=class_counts_path,
                level_key=level_key,
            )
            if len(counts) != self.num_classes:
                raise ValueError(
                    f"Expected {self.num_classes} class counts, got {len(counts)}."
                )
            weights = effective_number_weights(counts, beta=beta)
            # Optional explicit reweighting of the background class 0.
            if background_weight is not None:
                weights = weights.clone()
                weights[0] = float(background_weight)
                mean = weights.mean()
                if mean > 0:
                    weights = weights / mean
            self.register_buffer("class_weights", weights)
        else:
            self.register_buffer("class_weights", torch.ones(self.num_classes))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == logits.ndim:
            targets = targets.squeeze(1)
        return self.loss_weight * F.cross_entropy(
            logits,
            targets.long(),
            weight=self.class_weights.to(dtype=logits.dtype, device=logits.device),
            ignore_index=self.ignore_index,
        )


__all__ = [
    "EffectiveNumberCrossEntropyLoss",
    "effective_number_weights",
    "load_class_counts",
]

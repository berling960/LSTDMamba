"""Shared building blocks of the LSTD-Mamba network.

This module provides the primitives reused by every branch:

* :class:`ChannelMixer` -- the channel operator ``C`` of Eqs. (5), (6) and (10).
* :class:`GatedTemporalPooling` / :class:`GatedSpectralPooling` -- the learned
  reductions of Eqs. (7) and (11).
* :class:`SpatialDownsample` -- stride-2 spatial reduction inserted between
  adjacent encoder stages.
* :class:`ProgressiveOverlappingStem` -- the two overlapping 3x3 stride-2
  convolutions of Eq. (4).

``mamba_ssm`` is imported defensively: the registry stays importable on
machines without a CUDA Mamba build, and the error is raised only when a
Mamba-backed operator is actually constructed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # Keep registry discovery usable without the op.
    Mamba = None
    MAMBA_IMPORT_ERROR: ImportError | None = exc
else:
    MAMBA_IMPORT_ERROR = None


def group_count(channels: int, max_groups: int = 8) -> int:
    """Largest group count up to ``max_groups`` that divides ``channels``."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def channel_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """Apply LayerNorm over channels of a ``[B, T, C, H, W]`` tensor."""
    x = x.permute(0, 1, 3, 4, 2)
    x = norm(x)
    return x.permute(0, 1, 4, 2, 3).contiguous()


class DropPath(nn.Module):
    """Per-sample stochastic depth applied to residual branches."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask.div_(keep_prob)


class ChannelMixer(nn.Module):
    """Channel operator ``C``: MLP over the channel axis of ``[B,T,C,H,W]``."""

    def __init__(self, channels: int, norm_eps: float, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.SiLU(),
            nn.Linear(channels * mlp_ratio, channels),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.permute(0, 1, 3, 4, 2).contiguous()
        values = self.mlp(self.norm(values))
        return values.permute(0, 1, 4, 2, 3).contiguous()


class GatedTemporalPooling(nn.Module):
    """Learned temporal reduction (paper Eq. 7).

    A scalar gate produces one logit per time step and pixel; the softmax over
    ``t`` gives the normalized weights ``omega`` used to pool the sequence.
    This replaces uniform averaging, so the branch can emphasize the dates that
    are most informative for each location.
    """

    def __init__(self, channels: int, norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=norm_eps)
        self.gate = nn.Linear(channels, 1)
        # Zero-initialized so pooling starts as a uniform mean.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def pool_with_weights(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ``[B,T,C,H,W]`` to ``[B,C,H,W]`` and return the weights used."""
        values = x.permute(0, 1, 3, 4, 2).contiguous()
        logits = self.gate(self.norm(values))
        weights = torch.softmax(logits.float(), dim=1).to(dtype=values.dtype)
        pooled = (values * weights).sum(dim=1).permute(0, 3, 1, 2).contiguous()
        return pooled, weights.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pool_with_weights(x)
        return pooled


class GatedSpectralPooling(nn.Module):
    """Learned spectral reduction (paper Eq. 11).

    Reduces an ordered token sequence ``[N, L, C]`` to ``[N, C]`` with a
    normalized per-token importance weight. The identical operator serves both
    temporal and spectral aggregation; only the reduction axis differs.
    """

    def __init__(self, channels: int, norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=norm_eps)
        self.gate = nn.Linear(channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(self.norm(x))
        weights = torch.softmax(logits.float(), dim=1).to(dtype=x.dtype)
        return (x * weights).sum(dim=1)


class SpatialDownsample(nn.Module):
    """Stride-2 depthwise-separable reduction of H and W only.

    The acquisition-time axis is preserved, so the six observations remain
    ordered and available to every temporal operator.
    """

    def __init__(self, in_channels: int, out_channels: int, norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=norm_eps)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.depthwise.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.pointwise.weight, mode="fan_out", nonlinearity="relu"
        )
        if self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            raise ValueError(
                "SpatialDownsample requires even spatial dimensions, but "
                f"received H={x.shape[-2]}, W={x.shape[-1]}."
            )
        x = channel_norm(x, self.norm)
        batch, time, channels, height, width = x.shape
        x = x.reshape(batch * time, channels, height, width)
        x = self.pointwise(F.silu(self.depthwise(x)))
        out_height, out_width = x.shape[-2:]
        return x.reshape(batch, time, -1, out_height, out_width).contiguous()


class ProgressiveOverlappingStem(nn.Module):
    """Two overlapping 3x3 stride-2 convolutions (paper Eq. 4).

    The stem acts independently on each observation and preserves fine field
    boundaries before temporal aggregation::

        S = Stem(X_S) in R^{B x T x 64 x H/4 x W/4}
    """

    def __init__(self, in_channels: int, out_channels: int = 64) -> None:
        super().__init__()
        hidden_channels = max(out_channels // 2, in_channels)
        self.out_channels = int(out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"ProgressiveOverlappingStem expects [B, T, C, H, W], got "
                f"{tuple(x.shape)}."
            )
        batch, time, channels, height, width = x.shape
        x = x.reshape(batch * time, channels, height, width)
        x = self.layers(x)
        out_height, out_width = x.shape[-2:]
        return x.reshape(
            batch, time, self.out_channels, out_height, out_width
        ).contiguous()


__all__ = [
    "ChannelMixer",
    "DropPath",
    "GatedSpectralPooling",
    "GatedTemporalPooling",
    "Mamba",
    "MAMBA_IMPORT_ERROR",
    "ProgressiveOverlappingStem",
    "SpatialDownsample",
    "channel_norm",
    "group_count",
]

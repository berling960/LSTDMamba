"""Local Spatial-Temporal (LST) and Bidirectional Temporal Mamba (BTM) blocks.

Implements the Sentinel-2 temporal branch of LSTD-Mamba as described in
Section III-C of the paper.

The LST block is Eq. (5)::

    F'  = F  + S(F)        # depthwise spatial convolution
    F'' = F' + T(F')       # depthwise temporal convolution
    LST(F) = F'' + C(F'')  # channel mixer

The BTM block is Eq. (6)::

    F'  = F  + S(F)
    F'' = F' + BiMamba(F')  # bidirectional selective scan over acquisition time
    BTM(F) = F'' + C(F'')

Both blocks operate on ``[B, T, C, H, W]`` tensors so that the six
acquisition dates stay explicitly ordered and available to every temporal
operator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ChannelMixer,
    DropPath,
    Mamba,
    MAMBA_IMPORT_ERROR,
    channel_norm,
)


class LSTBlock(nn.Module):
    """Local spatial-temporal block (paper Eq. 5).

    Applies, in order: a depthwise spatial convolution that refines each
    observation independently, a depthwise temporal convolution that links
    neighbouring acquisition dates at every location, and a channel mixer.
    Layer normalization precedes every operator and stochastic depth is
    applied to each residual branch.
    """

    def __init__(
        self,
        channels: int,
        drop_path: float,
        norm_eps: float,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        use_spatial: bool = True,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        if spatial_kernel_size % 2 != 1:
            raise ValueError("spatial_kernel_size must be odd.")
        if temporal_kernel_size % 2 != 1:
            raise ValueError("temporal_kernel_size must be odd.")

        self.channels = int(channels)
        self.use_spatial = bool(use_spatial)
        self.use_temporal = bool(use_temporal)

        self.spatial_norm = (
            nn.LayerNorm(channels, eps=norm_eps) if self.use_spatial else None
        )
        self.spatial_mixer = (
            nn.Conv2d(
                channels,
                channels,
                kernel_size=spatial_kernel_size,
                padding=spatial_kernel_size // 2,
                groups=channels,
                bias=True,
            )
            if self.use_spatial
            else None
        )

        self.temporal_norm = (
            nn.LayerNorm(channels, eps=norm_eps) if self.use_temporal else None
        )
        self.temporal_mixer = (
            nn.Conv1d(
                channels,
                channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_kernel_size // 2,
                groups=channels,
                bias=True,
            )
            if self.use_temporal
            else None
        )

        self.channel_mixer = ChannelMixer(channels, norm_eps)
        self.drop_path = DropPath(drop_path)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mixer in (self.spatial_mixer, self.temporal_mixer):
            if mixer is None:
                continue
            nn.init.kaiming_normal_(
                mixer.weight, mode="fan_out", nonlinearity="relu"
            )
            if mixer.bias is not None:
                nn.init.zeros_(mixer.bias)

    def spatial_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """S(F): depthwise spatial convolution, applied per observation."""
        if self.spatial_norm is None or self.spatial_mixer is None:
            raise RuntimeError("The spatial operator is disabled.")
        y = channel_norm(x, self.spatial_norm)
        batch, time, channels, height, width = y.shape
        y = y.reshape(batch * time, channels, height, width)
        y = F.silu(self.spatial_mixer(y))
        return y.reshape(batch, time, channels, height, width).contiguous()

    def temporal_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """T(F): depthwise temporal convolution over the acquisition axis."""
        if self.temporal_norm is None or self.temporal_mixer is None:
            raise RuntimeError("The temporal operator is disabled.")
        y = channel_norm(x, self.temporal_norm)
        batch, time, channels, height, width = y.shape
        y = (
            y.permute(0, 3, 4, 2, 1)
            .reshape(batch * height * width, channels, time)
            .contiguous()
        )
        y = F.silu(self.temporal_mixer(y))
        return (
            y.reshape(batch, height, width, channels, time)
            .permute(0, 4, 3, 1, 2)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.channels:
            raise ValueError(
                f"LSTBlock expects [B, T, {self.channels}, H, W], "
                f"got {tuple(x.shape)}."
            )
        if self.use_spatial:
            x = x + self.drop_path(self.spatial_mixing(x))
        if self.use_temporal:
            x = x + self.drop_path(self.temporal_mixing(x))
        return x + self.drop_path(self.channel_mixer(x))


class BTMBlock(nn.Module):
    """Bidirectional Temporal Mamba block (paper Eq. 6).

    Keeps the spatial and channel operators of :class:`LSTBlock` and replaces
    only the local temporal convolution with :class:`BiMamba`, so local field
    structure is retained while dependencies are modelled across the complete
    acquisition sequence.
    """

    def __init__(
        self,
        channels: int,
        drop_path: float,
        norm_eps: float,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        bidirectional: bool = True,
        spatial_kernel_size: int = 3,
        use_spatial: bool = True,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        if spatial_kernel_size % 2 != 1:
            raise ValueError("spatial_kernel_size must be odd.")
        if use_temporal and Mamba is None:
            raise ImportError(
                "BTMBlock requires mamba_ssm.Mamba. Install a CUDA-compatible "
                "build with `pip install mamba-ssm`, or disable the temporal "
                "operator for CPU-only shape checks."
            ) from MAMBA_IMPORT_ERROR

        self.channels = int(channels)
        self.use_spatial = bool(use_spatial)
        self.use_temporal = bool(use_temporal)
        self.bidirectional = bool(bidirectional and self.use_temporal)

        self.spatial_norm = (
            nn.LayerNorm(channels, eps=norm_eps) if self.use_spatial else None
        )
        self.spatial_mixer = (
            nn.Conv2d(
                channels,
                channels,
                kernel_size=spatial_kernel_size,
                padding=spatial_kernel_size // 2,
                groups=channels,
                bias=True,
            )
            if self.use_spatial
            else None
        )

        self.temporal_norm = (
            nn.LayerNorm(channels, eps=norm_eps) if self.use_temporal else None
        )
        self.temporal_mamba = (
            Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
            if self.use_temporal
            else None
        )
        # Learnable direction scores; the mixing coefficient alpha of Eq. (3)
        # is their softmax, so the two scans are combined adaptively.
        if self.bidirectional:
            self.direction_logits = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("direction_logits", None)

        self.channel_mixer = ChannelMixer(channels, norm_eps)
        self.drop_path = DropPath(drop_path)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.spatial_mixer is not None:
            nn.init.kaiming_normal_(
                self.spatial_mixer.weight, mode="fan_out", nonlinearity="relu"
            )
            if self.spatial_mixer.bias is not None:
                nn.init.zeros_(self.spatial_mixer.bias)

    def direction_alpha(self) -> torch.Tensor:
        """Return the softmax direction weight of Eq. (3)."""
        if not self.bidirectional:
            raise RuntimeError("direction_alpha is only defined for bidirectional BTM.")
        return torch.softmax(self.direction_logits, dim=0)

    def spatial_mixing(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_norm is None or self.spatial_mixer is None:
            raise RuntimeError("The spatial operator is disabled.")
        y = channel_norm(x, self.spatial_norm)
        batch, time, channels, height, width = y.shape
        y = y.reshape(batch * time, channels, height, width)
        y = F.silu(self.spatial_mixer(y))
        return y.reshape(batch, time, channels, height, width).contiguous()

    def temporal_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """BiMamba(F'): shared-weight forward and reversed selective scans."""
        if self.temporal_norm is None or self.temporal_mamba is None:
            raise RuntimeError("The temporal operator is disabled.")
        y = channel_norm(x, self.temporal_norm)
        batch, time, channels, height, width = y.shape
        # One sequence per spatial position: [B*H*W, T, C].
        sequence = (
            y.permute(0, 3, 4, 1, 2)
            .reshape(batch * height * width, time, channels)
            .contiguous()
        )
        forward_sequence = self.temporal_mamba(sequence)
        if self.bidirectional:
            # The same Mamba parameters are reused for the reversed scan.
            reverse_sequence = self.temporal_mamba(
                torch.flip(sequence, dims=(1,)).contiguous()
            )
            reverse_sequence = torch.flip(
                reverse_sequence, dims=(1,)
            ).contiguous()
            weights = self.direction_alpha()
            sequence = (
                weights[0] * forward_sequence + weights[1] * reverse_sequence
            )
        else:
            sequence = forward_sequence
        return (
            sequence.reshape(batch, height, width, time, channels)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != self.channels:
            raise ValueError(
                f"BTMBlock expects [B, T, {self.channels}, H, W], "
                f"got {tuple(x.shape)}."
            )
        if self.use_spatial:
            x = x + self.drop_path(self.spatial_mixing(x))
        if self.use_temporal:
            x = x + self.drop_path(self.temporal_mixing(x))
        return x + self.drop_path(self.channel_mixer(x))


__all__ = ["LSTBlock", "BTMBlock"]

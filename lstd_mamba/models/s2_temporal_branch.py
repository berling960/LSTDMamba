"""Sentinel-2 temporal branch of LSTD-Mamba (paper Section III-C).

The encoder has three stages with channel widths ``(64, 128, 256)`` and one
block per stage. Spatial downsampling is inserted only between adjacent
stages, so the six observations remain ordered and available to every temporal
operator.

Following Algorithm 1, stages 1 and 2 use :class:`LSTBlock` because their
feature maps still contain detailed parcel boundaries, while the compact third
stage uses :class:`BTMBlock` to model dependencies across the complete
acquisition sequence. Every stage ends with the learned temporal pooling of
Eq. (7).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmseg.registry import MODELS

from .blocks import (
    GatedTemporalPooling,
    SpatialDownsample,
    ProgressiveOverlappingStem,
)
from .lst_btm import BTMBlock, LSTBlock


class LSTStage(nn.Module):
    """A stage of LST blocks followed by learned temporal pooling."""

    def __init__(
        self,
        channels: int,
        depth: int,
        drop_path_rates: Sequence[float],
        norm_eps: float,
    ) -> None:
        super().__init__()
        if len(drop_path_rates) != depth:
            raise ValueError(
                "drop_path_rates length must equal stage depth, but received "
                f"{len(drop_path_rates)} and {depth}."
            )
        self.blocks = nn.ModuleList(
            [
                LSTBlock(
                    channels=channels,
                    drop_path=drop_path_rates[index],
                    norm_eps=norm_eps,
                )
                for index in range(depth)
            ]
        )
        self.pool = GatedTemporalPooling(channels, norm_eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        return x, self.pool(x)


class BTMStage(nn.Module):
    """A stage of BTM blocks followed by learned temporal pooling."""

    def __init__(
        self,
        channels: int,
        depth: int,
        drop_path_rates: Sequence[float],
        norm_eps: float,
        d_state: int,
        d_conv: int,
        expand: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        if len(drop_path_rates) != depth:
            raise ValueError(
                "drop_path_rates length must equal stage depth, but received "
                f"{len(drop_path_rates)} and {depth}."
            )
        self.blocks = nn.ModuleList(
            [
                BTMBlock(
                    channels=channels,
                    drop_path=drop_path_rates[index],
                    norm_eps=norm_eps,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bidirectional=bidirectional,
                )
                for index in range(depth)
            ]
        )
        self.pool = GatedTemporalPooling(channels, norm_eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        return x, self.pool(x)


@MODELS.register_module()
class LSTDTimeEncoder(BaseModel):
    """Three-stage Sentinel-2 encoder with LST shallow stages and a BTM stage.

    Input ``[B, T, C_S, H, W]`` with ``T = 6`` observations. The output
    contract follows Algorithm 1: the three pooled stage features
    ``(F1_p, F2_p, F3_p)`` are returned so that the top-down decoder can
    combine them, and the pre-pooling temporal features are exposed for
    representation analysis.
    """

    def __init__(
        self,
        in_channels: int = 10,
        stage_channels: Sequence[int] = (64, 128, 256),
        depths: Sequence[int] = (1, 1, 1),
        btm_stage: int = 2,
        stem_channels: int = 64,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        drop_path_rate: float = 0.1,
        bidirectional: bool = True,
        norm_eps: float = 1e-6,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.in_channels = int(in_channels)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.depths = tuple(int(value) for value in depths)
        self.btm_stage = int(btm_stage)
        self.bidirectional = bool(bidirectional)
        self.norm_eps = float(norm_eps)

        if len(self.stage_channels) != 3:
            raise ValueError(
                "LSTDTimeEncoder requires exactly three stage_channels to "
                "produce F1, F2 and F3 as described in Algorithm 1."
            )
        if len(self.depths) != 3:
            raise ValueError("depths must contain exactly three values.")
        if any(value <= 0 for value in self.stage_channels + self.depths):
            raise ValueError("stage_channels and depths must be positive.")
        if self.btm_stage not in (0, 1, 2):
            raise ValueError(
                "btm_stage selects which stage uses BTM and must be 0, 1 or 2 "
                f"(the paper uses the final stage 2), got {self.btm_stage}."
            )
        # Stride-4 stem plus two stride-2 downsamplings.
        self.required_stride = 4 * 2 ** (len(self.stage_channels) - 1)

        self.stem = ProgressiveOverlappingStem(
            in_channels=self.in_channels, out_channels=stem_channels
        )

        all_drop_rates = torch.linspace(
            0, drop_path_rate, sum(self.depths)
        ).tolist()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        rate_offset = 0
        for index, (channels, depth) in enumerate(
            zip(self.stage_channels, self.depths)
        ):
            stage_rates = all_drop_rates[rate_offset : rate_offset + depth]
            if index == self.btm_stage:
                stage: nn.Module = BTMStage(
                    channels=channels,
                    depth=depth,
                    drop_path_rates=stage_rates,
                    norm_eps=norm_eps,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bidirectional=bidirectional,
                )
            else:
                stage = LSTStage(
                    channels=channels,
                    depth=depth,
                    drop_path_rates=stage_rates,
                    norm_eps=norm_eps,
                )
            self.stages.append(stage)
            rate_offset += depth

            if index < len(self.stage_channels) - 1:
                self.downsamples.append(
                    SpatialDownsample(
                        in_channels=channels,
                        out_channels=self.stage_channels[index + 1],
                        norm_eps=norm_eps,
                    )
                )

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                "LSTDTimeEncoder expects a torch.Tensor, but received "
                f"{type(inputs).__name__}."
            )
        if inputs.ndim == 4:
            raise ValueError(
                "LSTDTimeEncoder expects a five-dimensional tensor "
                "[B, T, C, H, W] describing the acquisition sequence, but "
                f"received {tuple(inputs.shape)}."
            )
        if inputs.ndim != 5:
            raise ValueError(
                "Expected input shape [B, T, C, H, W], but received "
                f"{tuple(inputs.shape)}."
            )
        if inputs.shape[2] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels at dimension 2, "
                f"but received shape {tuple(inputs.shape)}."
            )
        height, width = inputs.shape[-2:]
        if height % self.required_stride != 0 or width % self.required_stride != 0:
            raise ValueError(
                f"Input H and W must be divisible by {self.required_stride}, "
                f"but received H={height}, W={width}."
            )
        if not inputs.is_floating_point():
            raise TypeError(
                "LSTDTimeEncoder expects floating-point image data, but "
                f"received dtype {inputs.dtype}."
            )

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        del data_samples, mode
        self._validate_input(inputs)

        x = self.stem(inputs.contiguous())
        pooled_features: list[torch.Tensor] = []
        temporal_features: list[torch.Tensor] = []
        for index, stage in enumerate(self.stages):
            x, pooled = stage(x)
            temporal_features.append(x)
            pooled_features.append(pooled)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        return {
            # F3_p drives the decoder; F1_p/F2_p are the shallow skip features.
            "encoder_features": pooled_features[-1],
            "features_list": pooled_features[:-1],
            "stage_features": pooled_features,
            "distill_temporal_features": temporal_features,
            "ori_img": inputs,
        }


__all__ = ["LSTDTimeEncoder", "LSTStage", "BTMStage"]

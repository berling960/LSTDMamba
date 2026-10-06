"""Lightweight top-down fusion neck of LSTD-Mamba (paper Section III-B, III-E).

The decoder starts from the deepest pooled Sentinel-2 representation ``F3_p``,
progressively upsamples it, concatenates the shallower pooled maps, and applies
a pointwise projection after each combination. The multimodal fusion of
Eq. (13) is additive::

    F_M = Up( phi_F( F_S^d + Up( phi_P(F_E) ) ) )

The inner upsampling aligns the EnMAP feature with ``F_S^d``, and the outer
upsampling produces the prediction grid. Additive fusion keeps the alignment
step lightweight and avoids dense cross-modal interactions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

from .enmap_spectral_branch import SeparableProjection


class TopDownDecoder(nn.Module):
    """Upsample-and-concatenate decoder over the three pooled S2 stages.

    Implements step 12 of Algorithm 1. Starting from ``F3_p``, each step
    upsamples the running feature to the resolution of the next shallower map,
    concatenates it, and aligns the channels with a pointwise projection.
    """

    def __init__(
        self,
        stage_channels: Sequence[int] = (64, 128, 256),
        decoder_channels: Sequence[int] = (256, 128, 128),
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 3 or len(decoder_channels) != 3:
            raise ValueError(
                "stage_channels and decoder_channels must each contain exactly "
                "three values describing F1_p, F2_p and F3_p."
            )
        self.proj3 = nn.Conv2d(stage_channels[2], decoder_channels[0], 1)
        self.fuse2 = SeparableProjection(
            decoder_channels[0] + stage_channels[1], decoder_channels[1]
        )
        self.fuse1 = SeparableProjection(
            decoder_channels[1] + stage_channels[0], decoder_channels[2]
        )
        self.out = nn.Conv2d(decoder_channels[2], out_channels, 1)
        for layer in (self.proj3, self.out):
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu"
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        stage_features: Sequence[torch.Tensor],
        out_size: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if len(stage_features) != 3:
            raise ValueError(
                "TopDownDecoder expects the three pooled stage features "
                f"(F1_p, F2_p, F3_p), got {len(stage_features)}."
            )
        f1, f2, f3 = stage_features
        x = self.proj3(f3)
        x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([x, f1], dim=1))
        x = self.out(x)
        if out_size is not None:
            x = F.interpolate(
                x, size=tuple(out_size), mode="bilinear", align_corners=False
            )
        return x


@MODELS.register_module()
class LSTDFusionNeck(nn.Module):
    """Decode Sentinel-2 stages and fuse the EnMAP feature additively.

    ``in_feature_key`` selects which encoder outputs feed the decoder. When an
    EnMAP key is present the spectral feature is projected to the Sentinel-2
    channel width by ``phi_P`` and fused additively per Eq. (13); the fused
    feature is then upsampled to the prediction grid.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        in_feature_key: Sequence[str] = ("S2",),
        feature_size: Sequence[int] = (48, 48),
        out_size: Sequence[int] = (192, 192),
        fusion_channels: Sequence[int] = (256, 128, 128),
        out_channels: int = 128,
        stage_channels: Sequence[int] = (64, 128, 256),
        hyper_embed_neck: dict | None = None,
    ) -> None:
        super().__init__()
        self.in_feature_key = tuple(in_feature_key)
        self.out_size = tuple(int(value) for value in out_size)
        self.feature_size = tuple(int(value) for value in feature_size)
        self.out_channels = int(out_channels)

        self.decoder = TopDownDecoder(
            stage_channels=stage_channels,
            decoder_channels=fusion_channels,
            out_channels=out_channels,
        )
        self.has_hyper = hyper_embed_neck is not None
        if self.has_hyper:
            if not isinstance(hyper_embed_neck, Mapping):
                raise TypeError(
                    "hyper_embed_neck must be a configuration mapping, got "
                    f"{type(hyper_embed_neck).__name__}."
                )
            cfg = dict(hyper_embed_neck)
            # phi_P: project the EnMAP feature to the S2 channel width.
            self.hyper_projection = SeparableProjection(
                int(cfg["in_channels"]), out_channels
            )
            self.hyper_key = str(cfg.get("in_key", "EnMAP"))

    def _stage_features(self, inputs: Mapping[str, object]) -> list[torch.Tensor]:
        encoder_outputs = inputs
        for key in self.in_feature_key:
            if key not in encoder_outputs:
                raise KeyError(
                    f"Expected encoder output {key!r}, available keys: "
                    f"{sorted(encoder_outputs)}."
                )
            encoder_outputs = encoder_outputs[key]
        if not isinstance(encoder_outputs, Mapping):
            raise TypeError(
                "Encoder output for the fusion neck must be a mapping."
            )

        if "stage_features" in encoder_outputs:
            stage_features = list(encoder_outputs["stage_features"])
        else:
            deepest = encoder_outputs.get("encoder_features")
            shallow = list(encoder_outputs.get("features_list", []))
            if deepest is None or len(shallow) != 2:
                raise KeyError(
                    "Expected either 'stage_features' or "
                    "'encoder_features' plus a two-element 'features_list'."
                )
            stage_features = shallow + [deepest]

        if len(stage_features) != 3:
            raise ValueError(
                "The fusion neck expects three pooled Sentinel-2 stages, got "
                f"{len(stage_features)}."
            )
        return stage_features

    def forward(self, inputs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        stage_features = self._stage_features(inputs)
        decoded = self.decoder(stage_features, out_size=self.out_size)

        if self.has_hyper:
            if self.hyper_key not in inputs:
                raise KeyError(
                    f"hyper_fusion requires inputs[{self.hyper_key!r}]."
                )
            hyper_feature = inputs[self.hyper_key]
            if not isinstance(hyper_feature, Mapping) or "ori_feature" not in hyper_feature:
                raise KeyError(
                    f"Expected inputs[{self.hyper_key!r}]['ori_feature'] for "
                    "the EnMAP branch."
                )
            hyper_feature = hyper_feature["ori_feature"]
            # phi_P followed by the inner upsampling of Eq. (13).
            aligned = F.interpolate(
                self.hyper_projection(hyper_feature),
                size=decoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = decoded + aligned
        else:
            fused = decoded

        return {
            "fused_feature": fused,
            "s2_decoded": decoded,
            "stage_features": stage_features,
        }


__all__ = ["LSTDFusionNeck", "TopDownDecoder"]

"""LSTD-Mamba: the complete deployable network (paper Algorithm 1).

This module assembles the four components described in Section III:

1. :class:`~lstd_mamba.models.s2_temporal_branch.LSTDTimeEncoder` -- the
   Sentinel-2 temporal branch with LST shallow stages and a BTM deep stage.
2. :class:`~lstd_mamba.models.enmap_spectral_branch.LSTDSpectralEncoder` --
   the EnMAP spectral branch with a full-resolution local path and BSM.
3. :class:`~lstd_mamba.models.fusion_neck.LSTDFusionNeck` -- the lightweight
   top-down decoder and additive multimodal fusion.
4. :class:`~lstd_mamba.models.cascade_head.PriorAwareCascadeHead` -- the
   prior-aware four-level cascade head.

The same deployable network is optimized and evaluated, so the parameter and
FLOP counts describe the complete forward path: the two modality branches, the
lightweight fusion operators, and the four supervised heads.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from mmseg.registry import MODELS

from .cascade_head import DEFAULT_LEVELS, PriorAwareCascadeHead
from .enmap_spectral_branch import LSTDSpectralEncoder
from .fusion_neck import LSTDFusionNeck
from .losses import EffectiveNumberCrossEntropyLoss
from .s2_temporal_branch import LSTDTimeEncoder


@MODELS.register_module()
class LSTDMamba(nn.Module):
    """Full LSTD-Mamba model producing four hierarchical logit maps.

    In ``"loss"`` mode the model returns ``(outputs, losses)`` where ``losses``
    contains the per-level weighted cross-entropies and their sum, matching
    Eq. (16).
    """

    def __init__(
        self,
        s2_encoder: Optional[dict] = None,
        enmap_encoder: Optional[dict] = None,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        levels: Optional[tuple] = None,
        with_priors: bool = True,
        with_enmap: bool = True,
        s2_key: str = "S2",
        enmap_key: str = "EnMAP",
        prior_key: str = "priors",
        loss: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.s2_key = str(s2_key)
        self.enmap_key = str(enmap_key)
        self.prior_key = str(prior_key)
        self.with_priors = bool(with_priors)
        self.with_enmap = bool(with_enmap)

        level_cfg = tuple(levels) if levels is not None else DEFAULT_LEVELS
        self.level_names = [str(entry[0]) for entry in level_cfg]

        s2_cfg = dict(s2_encoder) if s2_encoder else {}
        self.s2_encoder = MODELS.build({"type": "LSTDTimeEncoder", **s2_cfg})

        self.enmap_encoder = None
        if self.with_enmap:
            enmap_cfg = dict(enmap_encoder) if enmap_encoder else {}
            self.enmap_encoder = MODELS.build(
                {"type": "LSTDSpectralEncoder", **enmap_cfg}
            )

        neck_cfg = dict(neck) if neck else {}
        self.neck = MODELS.build({"type": "LSTDFusionNeck", **neck_cfg})

        head_cfg = dict(head) if head else {}
        head_cfg.setdefault("levels", level_cfg)
        head_cfg.setdefault("with_priors", self.with_priors)
        self.head = MODELS.build({"type": "PriorAwareCascadeHead", **head_cfg})

        # Eq. (16): one effective-number weighted cross-entropy per level.
        self.loss_functions = None
        if loss is not None:
            self.loss_functions = self.build_loss_functions(loss, level_cfg)

    def build_loss_functions(
        self,
        loss: dict,
        levels: Optional[tuple] = None,
    ) -> nn.ModuleDict:
        """Build one :class:`EffectiveNumberCrossEntropyLoss` per level.

        ``loss`` accepts either a shared configuration applied to all four
        levels, or a mapping from level name to its own configuration.
        """
        level_cfg = levels if levels is not None else tuple(
            zip(self.level_names, [7, 37, 83, 102])
        )
        per_level: dict[str, nn.Module] = {}
        for name, num_classes in level_cfg:
            cfg = dict(loss)
            if all(isinstance(value, dict) for value in loss.values()):
                cfg = dict(loss.get(str(name), loss))
            cfg.pop("type", None)
            cfg["num_classes"] = int(num_classes)
            cfg.setdefault("level_key", str(name))
            per_level[str(name)] = EffectiveNumberCrossEntropyLoss(**cfg)
        return nn.ModuleDict(per_level)


    def _historical_priors(
        self, data_samples: Optional[object]
    ) -> Optional[dict[str, torch.Tensor]]:
        """Extract the per-level historical prior maps from the data samples."""
        if not self.with_priors or data_samples is None:
            return None
        if isinstance(data_samples, dict):
            return data_samples
        priors: dict[str, torch.Tensor] = {}
        for name in self.level_names:
            value = None
            if hasattr(data_samples, "get"):
                value = data_samples.get(name)
            if value is None and hasattr(data_samples, name):
                value = getattr(data_samples, name)
            if value is not None:
                priors[name] = value
        return priors or None

    def forward(
        self,
        inputs: dict,
        data_samples: Optional[object] = None,
        mode: str = "tensor",
    ) -> Union[dict[str, torch.Tensor], list, tuple]:
        if self.s2_key not in inputs:
            raise KeyError(
                f"LSTD-Mamba requires inputs[{self.s2_key!r}] for the "
                "Sentinel-2 temporal branch."
            )

        encoder_outputs: dict[str, object] = {
            self.s2_key: self.s2_encoder(inputs[self.s2_key], mode="tensor")
        }
        if self.enmap_encoder is not None:
            if self.enmap_key not in inputs:
                raise KeyError(
                    f"with_enmap=True requires inputs[{self.enmap_key!r}]."
                )
            encoder_outputs[self.enmap_key] = self.enmap_encoder(
                inputs[self.enmap_key], mode="tensor"
            )

        fused = self.neck(encoder_outputs)["fused_feature"]
        priors = self._historical_priors(data_samples)
        logits = self.head(fused, priors)

        if mode in {"tensor", "predict"}:
            return [logits]

        if mode != "loss":
            raise ValueError(
                f"Unsupported mode {mode!r}; expected 'tensor', 'predict' or 'loss'."
            )
        targets = self._targets(data_samples)
        losses = self._level_losses(logits, targets)
        return logits, losses

    @staticmethod
    def _targets(data_samples: Optional[object]) -> dict[str, torch.Tensor]:
        if data_samples is None:
            raise ValueError("data_samples are required in loss mode.")
        targets: dict[str, torch.Tensor] = {}
        if isinstance(data_samples, dict):
            for key, value in data_samples.items():
                if isinstance(key, str) and key.startswith("level"):
                    targets[key] = value
        else:
            for name in ("level1", "level2", "level3", "level4"):
                if hasattr(data_samples, name):
                    targets[name] = getattr(data_samples, name)
        if not targets:
            raise ValueError(
                "No level targets found in data_samples; expected keys named "
                "'level1'..'level4'."
            )
        return targets

    def _level_losses(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sum the four weighted cross-entropies of Eq. (16)."""
        if self.loss_functions is None:
            raise RuntimeError(
                "build_loss_functions() must be called before loss mode is used."
            )
        losses: dict[str, torch.Tensor] = {}
        total: Optional[torch.Tensor] = None
        for name in self.level_names:
            if name not in targets:
                raise KeyError(
                    f"Missing target for level {name!r}; available keys: "
                    f"{sorted(targets)}."
                )
            value = self.loss_functions[name](logits[name], targets[name])
            losses[f"loss_{name}"] = value
            total = value if total is None else total + value
        losses["loss_task"] = total
        return losses


__all__ = ["LSTDMamba"]

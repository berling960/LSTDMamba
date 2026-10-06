"""Prior-aware four-level cascade classification head (paper Section III-E-2).

Each level has its own head. Level 1 receives the fused representation and its
historical prior; every finer level additionally receives the continuous logits
of the preceding level, as in Eq. (14)::

    O_1 = H_1(Cat(F_M, Pi_1))
    O_l = H_l(Cat(F_M, Pi_l, O_{l-1})),  l = 2, 3, 4

Each ``H_l`` contains a 3x3 convolution, a ReLU activation and a 1x1 classifier
projection. Continuous logits are passed rather than hard parent labels so that
coarse category evidence is preserved while finer predictions stay
differentiable. The label levels contain 7, 37, 83 and 102 classes.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

from .blocks import group_count


#: Level names and their class counts, as used by H2Crop (including background).
DEFAULT_LEVELS: tuple[tuple[str, int], ...] = (
    ("level1", 7),
    ("level2", 37),
    ("level3", 83),
    ("level4", 102),
)


class CascadeLevelHead(nn.Module):
    """One level of the cascade: 3x3 convolution, ReLU, then 1x1 classifier."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_classes: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.norm = nn.GroupNorm(group_count(in_channels), in_channels, eps=norm_eps)
        self.conv = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, padding=1, bias=False
        )
        self.act = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.normal_(self.classifier.weight, std=0.01)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.act(self.conv(self.norm(x))))


@MODELS.register_module()
class PriorAwareCascadeHead(nn.Module):
    """Four nested classifiers driven by priors and coarse-to-fine logits.

    Historical prior maps ``R_l`` are supplied as integer label maps and are
    converted to one-hot features ``Pi_l`` before being concatenated with the
    fused representation.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        levels: Optional[Sequence[Sequence[int | str]]] = None,
        with_priors: bool = True,
        prior_ignore_index: int = 255,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.with_priors = bool(with_priors)
        self.prior_ignore_index = int(prior_ignore_index)

        level_cfg = levels if levels is not None else DEFAULT_LEVELS
        self.level_names: list[str] = []
        self.num_classes: list[int] = []
        for entry in level_cfg:
            name, count = entry[0], int(entry[1])
            self.level_names.append(str(name))
            self.num_classes.append(count)

        self.heads = nn.ModuleDict()
        # Level 1 consumes F_M; each finer level also consumes the previous logits.
        in_channels = embed_dim
        for index, (name, count) in enumerate(
            zip(self.level_names, self.num_classes)
        ):
            level_in = in_channels + (count if self.with_priors else 0)
            self.heads[name] = CascadeLevelHead(
                in_channels=level_in,
                embed_dim=embed_dim,
                num_classes=count,
                norm_eps=norm_eps,
            )
            in_channels = embed_dim + count

    def _one_hot_prior(self, prior: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert an integer prior map to a one-hot feature map."""
        if prior.ndim == 4:
            prior = prior.squeeze(1)
        valid = prior != self.prior_ignore_index
        safe = torch.where(valid, prior, torch.zeros_like(prior))
        one_hot = F.one_hot(safe.long(), num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=torch.float32)
        return one_hot * valid.unsqueeze(1).to(dtype=torch.float32)

    def forward(
        self,
        fused_feature: torch.Tensor,
        priors: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Run the cascade and return the four logit maps."""
        if fused_feature.ndim != 4:
            raise ValueError(
                "PriorAwareCascadeHead expects [B, C, H, W], got "
                f"{tuple(fused_feature.shape)}."
            )
        if self.with_priors and priors is None:
            raise ValueError(
                "with_priors=True requires a priors mapping keyed by level name."
            )

        outputs: dict[str, torch.Tensor] = {}
        previous_logits: Optional[torch.Tensor] = None
        for name, count in zip(self.level_names, self.num_classes):
            inputs = [fused_feature]
            if self.with_priors:
                if name not in priors:
                    raise KeyError(
                        f"Missing historical prior for level {name!r}; "
                        f"available keys: {sorted(priors)}."
                    )
                if previous_logits is not None:
                    # Coarse logits are resized if a level changes resolution.
                    if previous_logits.shape[-2:] != fused_feature.shape[-2:]:
                        previous_logits = F.interpolate(
                            previous_logits,
                            size=fused_feature.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    inputs.append(previous_logits)
                inputs.append(self._one_hot_prior(priors[name], count))
            logits = self.heads[name](torch.cat(inputs, dim=1))
            outputs[name] = logits
            previous_logits = logits
        return outputs


__all__ = ["PriorAwareCascadeHead", "CascadeLevelHead", "DEFAULT_LEVELS"]

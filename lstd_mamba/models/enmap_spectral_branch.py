"""EnMAP spectral branch of LSTD-Mamba (paper Section III-D).

The branch separates local spatial projection from spectral sequence
modelling, so the expensive spectral scan is never repeated at full spatial
resolution:

* **Local projection** (Eq. 8) applies a 1x1 projection to the original
  resolution cube and preserves boundary information that pooling would weaken.
* **BSM path** (Eqs. 9-11) average-pools the spatial grid with stride four,
  converts each pooled spectrum into an ordered token sequence, adds a
  learnable band embedding, and processes it with two bidirectional spectral
  Mamba blocks.
* **Fusion** (Eq. 12) concatenates both paths and projects them back to the
  original spatial grid.

Because the token representation is only 32 channels wide, the full-resolution
scan is avoided while global band relations are still captured.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmseg.registry import MODELS

from .blocks import (
    ChannelMixer,
    DropPath,
    GatedSpectralPooling,
    Mamba,
    MAMBA_IMPORT_ERROR,
    group_count,
)


class BSMTokenizer(nn.Module):
    """Average pooling plus 1-D convolution and band embedding (Eq. 9).

    Produces ``F0_E in R^{(B H' W') x L x d}``: one ordered spectral sequence
    per pooled spatial location. The same tokenization is applied independently
    at every pooled location.
    """

    def __init__(
        self,
        in_bands: int,
        token_channels: int,
        spatial_pool: int = 4,
        spectral_stride: int = 2,
    ) -> None:
        super().__init__()
        if spatial_pool <= 1:
            raise ValueError(f"spatial_pool must exceed 1, got {spatial_pool}.")
        if spectral_stride <= 0:
            raise ValueError(
                f"spectral_stride must be positive, got {spectral_stride}."
            )
        self.in_bands = int(in_bands)
        self.token_channels = int(token_channels)
        self.spatial_pool = int(spatial_pool)
        self.spectral_stride = int(spectral_stride)
        # L is the token count after the strided 1-D convolution.
        self.token_count = (
            self.in_bands - self.spectral_stride
        ) // self.spectral_stride + 1
        if self.token_count <= 0:
            raise ValueError(
                f"spectral_stride={self.spectral_stride} removes all "
                f"{self.in_bands} bands; choose a smaller stride."
            )

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=self.token_channels,
            kernel_size=self.spectral_stride,
            stride=self.spectral_stride,
            bias=False,
        )
        self.norm = nn.LayerNorm(self.token_channels)
        # Learnable band embedding e in R^{L x d}, added to every sequence.
        self.band_embedding = nn.Parameter(
            torch.zeros(1, self.token_count, self.token_channels)
        )
        nn.init.trunc_normal_(self.band_embedding, std=0.02)
        nn.init.kaiming_normal_(
            self.conv1d.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_bands:
            raise ValueError(
                f"BSMTokenizer expects [B, {self.in_bands}, H, W], got "
                f"{tuple(inputs.shape)}."
            )
        # Spatial pooling keeps the scan on a compact grid.
        pooled = F.avg_pool2d(
            inputs, kernel_size=self.spatial_pool, stride=self.spatial_pool
        )
        batch, _, height, width = pooled.shape
        # One sequence per pooled spatial location.
        sequences = pooled.reshape(batch, self.in_bands, height * width).permute(
            0, 2, 1
        )
        sequences = sequences.reshape(-1, 1, self.in_bands)
        tokens = F.silu(self.norm(self.conv1d(sequences).transpose(1, 2)))
        return tokens + self.band_embedding


class BSMBlock(nn.Module):
    """Bidirectional Spectral Mamba block (paper Eq. 10).

    ``F'  = F + BiMamba(LN(F))`` and ``BSM(F) = F' + C(LN(F'))``. The two scans
    access the ordered spectral response from opposite directions, and the
    channel operator then refines the token channels. The same Mamba parameters
    are shared by both directions, as in Eq. (3).
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        drop_path: float = 0.0,
        bidirectional: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "BSMBlock requires mamba_ssm.Mamba. Install a CUDA-compatible "
                "build with `pip install mamba-ssm`."
            ) from MAMBA_IMPORT_ERROR
        self.channels = int(channels)
        self.bidirectional = bool(bidirectional)
        self.spectral_norm = nn.LayerNorm(channels, eps=norm_eps)
        self.spectral_mamba = Mamba(
            d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand
        )
        if self.bidirectional:
            self.direction_logits = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("direction_logits", None)
        self.channel_norm = nn.LayerNorm(channels, eps=norm_eps)
        self.channel_mixer = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        for layer in self.channel_mixer:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.drop_path = DropPath(drop_path)

    def direction_alpha(self) -> torch.Tensor:
        """Return the softmax direction weight of Eq. (3)."""
        if not self.bidirectional:
            raise RuntimeError("direction_alpha is only defined for bidirectional BSM.")
        return torch.softmax(self.direction_logits, dim=0)

    def bi_mamba(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.spectral_norm(x)
        forward_sequence = self.spectral_mamba(sequence)
        if not self.bidirectional:
            return forward_sequence
        reverse_sequence = self.spectral_mamba(
            torch.flip(sequence, dims=(1,)).contiguous()
        )
        reverse_sequence = torch.flip(reverse_sequence, dims=(1,)).contiguous()
        weights = self.direction_alpha()
        return weights[0] * forward_sequence + weights[1] * reverse_sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"BSMBlock expects [N, L, {self.channels}], got "
                f"{tuple(x.shape)}."
            )
        x = x + self.drop_path(self.bi_mamba(x))
        x = x + self.drop_path(self.channel_mixer(self.channel_norm(x)))
        return x


class SeparableProjection(nn.Module):
    """1x1 convolution, group normalization and SiLU, used for phi_E / phi_S."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(group_count(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


@MODELS.register_module()
class LSTDSpectralEncoder(BaseModel):
    """EnMAP encoder combining a full-resolution local path with BSM.

    Parameters follow Section IV-A: ``C = 218`` bands at ``64 x 64`` for the
    input, a 32-channel token representation, and two BSM blocks. The returned
    ``ori_feature`` is the full-resolution fused EnMAP feature ``F_E`` of
    Eq. (12).
    """

    def __init__(
        self,
        in_bands: int = 218,
        spectral_channels: int = 32,
        out_channels: int = 64,
        depth: int = 2,
        spatial_pool: int = 4,
        spectral_stride: int = 2,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
        drop_path_rate: float = 0.05,
        bidirectional: bool = True,
        branch_mode: str = "dual",
        local_branch_drop_prob: float = 0.0,
        fusion_gate_bias: float = -1.0,
        norm_eps: float = 1e-6,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if branch_mode not in {"dual", "local_only", "spectral_only"}:
            raise ValueError(
                "branch_mode must be 'dual', 'local_only' or 'spectral_only', "
                f"got {branch_mode!r}."
            )
        if branch_mode != "local_only" and Mamba is None:
            raise ImportError(
                "LSTDSpectralEncoder requires mamba_ssm.Mamba unless "
                "branch_mode='local_only'."
            ) from MAMBA_IMPORT_ERROR

        self.in_bands = int(in_bands)
        self.spectral_channels = int(spectral_channels)
        self.out_channels = int(out_channels)
        self.depth = int(depth)
        self.branch_mode = branch_mode
        self.local_branch_drop_prob = float(local_branch_drop_prob)
        self.spatial_pool = int(spatial_pool)

        self.local_project = (
            nn.Conv2d(in_bands, out_channels, kernel_size=1, bias=False)
            if branch_mode != "spectral_only"
            else None
        )

        if branch_mode != "local_only":
            self.tokenizer = BSMTokenizer(
                in_bands=in_bands,
                token_channels=spectral_channels,
                spatial_pool=spatial_pool,
                spectral_stride=spectral_stride,
            )
            self.bsm_blocks = nn.ModuleList(
                [
                    BSMBlock(
                        channels=spectral_channels,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        drop_path=drop_path_rate,
                        bidirectional=bidirectional,
                        norm_eps=norm_eps,
                    )
                    for _ in range(depth)
                ]
            )
            self.spectral_pool = GatedSpectralPooling(
                spectral_channels, norm_eps
            )
            self.spectral_project = nn.Conv2d(
                spectral_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.tokenizer = None
            self.bsm_blocks = nn.ModuleList()
            self.spectral_pool = None
            self.spectral_project = None

        if branch_mode == "dual":
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.fusion_gate[0].weight)
            nn.init.constant_(self.fusion_gate[0].bias, fusion_gate_bias)

        self.local_norm = nn.GroupNorm(group_count(out_channels), out_channels)

    def _validate_input(self, inputs: torch.Tensor) -> None:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                "LSTDSpectralEncoder expects a torch.Tensor, but received "
                f"{type(inputs).__name__}."
            )
        if inputs.ndim != 4:
            raise ValueError(
                "LSTDSpectralEncoder expects [B, C, H, W], but received "
                f"{tuple(inputs.shape)}."
            )
        if inputs.shape[1] != self.in_bands:
            raise ValueError(
                f"Expected {self.in_bands} EnMAP bands, but received "
                f"{inputs.shape[1]}."
            )
        if not inputs.is_floating_point():
            raise TypeError(
                "LSTDSpectralEncoder expects floating-point data, but received "
                f"dtype {inputs.dtype}."
            )

    def _spectral_path(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(inputs)
        for block in self.bsm_blocks:
            tokens = block(tokens)
        pooled = self.spectral_pool(tokens)
        batch = inputs.shape[0]
        height = inputs.shape[-2] // self.spatial_pool
        width = inputs.shape[-1] // self.spatial_pool
        feature = pooled.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        return self.spectral_project(feature.contiguous())

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[list] = None,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor]:
        del data_samples, mode
        self._validate_input(inputs)

        if self.branch_mode == "local_only":
            feature = F.silu(self.local_norm(self.local_project(inputs)))
            return {"ori_feature": feature, "spectral_gate_mean": feature.new_zeros(1)}

        local_feature = F.silu(self.local_norm(self.local_project(inputs)))
        spectral_feature = self._spectral_path(inputs)
        # Eq. (12): upsample the pooled spectral path back to the local grid.
        spectral_feature = F.interpolate(
            spectral_feature,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.branch_mode == "spectral_only":
            feature = spectral_feature
            gate_mean = feature.new_ones(1)
        else:
            gate = self.fusion_gate(
                torch.cat([local_feature, spectral_feature], dim=1)
            )
            # Gated addition: the full-resolution path is the anchor.
            feature = local_feature + gate * spectral_feature
            gate_mean = gate.mean().detach().reshape(1)

        return {"ori_feature": feature, "spectral_gate_mean": gate_mean}


__all__ = ["LSTDSpectralEncoder", "BSMBlock", "BSMTokenizer"]

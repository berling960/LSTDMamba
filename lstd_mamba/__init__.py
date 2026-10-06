"""LSTD-Mamba: a lightweight spectral-temporal dual-Mamba network.

LSTD-Mamba performs multimodal hierarchical crop classification from
multitemporal Sentinel-2 observations, EnMAP hyperspectral imagery, and
historical crop priors.

The network has four parts, all implemented in :mod:`lstd_mamba.models`:

* a **Sentinel-2 temporal branch** built from Local Spatial-Temporal (LST)
  blocks in the shallow stages and a Bidirectional Temporal Mamba (BTM) block
  in the compact final stage;
* an **EnMAP spectral branch** combining a full-resolution local projection
  with a Bidirectional Spectral Mamba (BSM) path on a spatially pooled grid;
* a **lightweight top-down fusion neck** with additive multimodal fusion;
* a **prior-aware cascade head** predicting four nested label levels
  (7, 37, 83 and 102 classes on H2Crop).

The dataset pipeline and training engine come from the upstream H2Crop
project, which must be installed separately.

Reference
---------
LSTD-Mamba: A Lightweight Spectral-Temporal Dual-Mamba Network for Multimodal
Hierarchical Crop Classification.
"""

from .models import (
    BSMBlock,
    BSMTokenizer,
    BTMBlock,
    CascadeLevelHead,
    EffectiveNumberCrossEntropyLoss,
    LSTBlock,
    LSTDFusionNeck,
    LSTDMamba,
    LSTDSpectralEncoder,
    LSTDTimeEncoder,
    PriorAwareCascadeHead,
    TopDownDecoder,
    effective_number_weights,
)

__version__ = "0.1.0"

__all__ = [
    "LSTDMamba",
    "LSTDTimeEncoder",
    "LSTDSpectralEncoder",
    "LSTDFusionNeck",
    "PriorAwareCascadeHead",
    "LSTBlock",
    "BTMBlock",
    "BSMBlock",
    "BSMTokenizer",
    "TopDownDecoder",
    "CascadeLevelHead",
    "EffectiveNumberCrossEntropyLoss",
    "effective_number_weights",
    "__version__",
]

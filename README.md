<div align="center">

# LSTD-Mamba

**A Lightweight Spectral–Temporal Dual-Mamba Network for Multimodal Hierarchical Crop Classification**

</div>

Official code release for **LSTD-Mamba**, a lightweight network for
fine-grained hierarchical crop type classification from multitemporal
Sentinel-2 observations, EnMAP hyperspectral imagery, and historical crop
priors.

LSTD-Mamba models dependencies along the acquisition-time and spectral-band
axes without a costly joint-attention module:

- **Sentinel-2 stream** — local spatial–temporal blocks in the shallow stages,
  a bidirectional temporal Mamba block in the final stage.
- **EnMAP stream** — a full-resolution local projection combined with a
  bidirectional spectral Mamba path on a spatially pooled grid.
- **Fusion neck** — lightweight top-down fusion of the two streams.
- **Prior-aware cascade head** — predicts four nested label levels.

## Installation

LSTD-Mamba reuses the dataset pipeline and training engine of the upstream
**H2Crop** project, which is *not* vendored here.

```bash
conda create -n lstd python=3.10 -y
conda activate lstd

# PyTorch matching your CUDA driver, first.
pip install torch --index-url https://download.pytorch.org/whl/cu124

# LSTD-Mamba
git clone <LSTD_MAMBA_REPO_URL>
cd LSTD-Mamba && pip install -e .

# Mamba kernels (needs a working nvcc toolchain)
pip install mamba-ssm>=2.2.2

# Upstream H2Crop
git clone <H2CROP_REPO_URL> ../H2Crop-main
export H2CROP_ROOT=$(realpath ../H2Crop-main)
```

`tools/_paths.py` also finds H2Crop automatically in `./H2Crop-main` or
`../H2Crop-main`, so the environment variable is often unnecessary.

## Usage

Prepare the H2Crop data, then train and evaluate:

```bash
# Train (2 GPUs)
torchrun --standalone --nproc_per_node=2 tools/train.py \
    configs/lstd_mamba.py --launcher pytorch --work-dir work_dirs/lstd_mamba

# Single-process
python tools/train.py configs/lstd_mamba.py --work-dir work_dirs/lstd_mamba

# Evaluate a checkpoint
python tools/test.py configs/lstd_mamba.py \
    work_dirs/lstd_mamba/best_mFscore_epoch_XX.pth
```

Check that a configuration builds and see its parameter budget:

```bash
python tools/check_model.py --build-only
```

`tools/smoke_test.py` verifies the tensor shapes of both branches and needs no
H2Crop checkout and no checkpoint:

```bash
python tools/smoke_test.py --device cuda
```

Components can also be used directly:

```python
import torch
from lstd_mamba import LSTDTimeEncoder, LSTDSpectralEncoder

s2 = LSTDTimeEncoder(stage_channels=(64, 128, 256), depths=(1, 1, 1), btm_stage=2)
print([tuple(f.shape) for f in s2(torch.randn(1, 6, 10, 192, 192))["stage_features"]])

enmap = LSTDSpectralEncoder(in_bands=218, spectral_channels=32, out_channels=64)
print(enmap(torch.randn(1, 218, 64, 64))["ori_feature"].shape)
```

## Repository layout

```text
LSTD-Mamba/
├── lstd_mamba/models/            # branches, blocks, fusion neck, cascade head
├── configs/lstd_mamba.py         # reference (student-only) configuration
└── tools/                        # train / test / check / smoke_test
```

Code comments cite the corresponding equations and sections of the paper.

## Citation

```bibtex
@article{lstdmamba,
  title   = {LSTD-Mamba: A Lightweight Spectral-Temporal Dual-Mamba Network
             for Multimodal Hierarchical Crop Classification},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

## License

Apache License 2.0. This repository contains only the LSTD-Mamba model code;
the upstream H2Crop project, `mamba-ssm`, `mmcv`, `mmengine` and PyTorch remain
under their own licenses.

## Acknowledgements

Built on the [H2Crop](https://github.com/lzr-2024/H2Crop) dataset and codebase,
the [MMEngine](https://github.com/open-mmlab/mmengine) /
[MMCV](https://github.com/open-mmlab/mmcv) registries, and
[mamba-ssm](https://github.com/state-spaces/mamba).

"""Shape and parameter-count self-test for the LSTD-Mamba branches.

Usage:
    python tools/smoke_test.py --device cuda

This script exercises only ``lstd_mamba``, so it needs no H2Crop checkout and
no checkpoint. It verifies the tensor contract of the paper's Algorithm 1:

* the Sentinel-2 branch maps ``[B, 6, 10, 192, 192]`` to pooled stage features
  at 48x48, 24x24 and 12x12 with widths 64, 128 and 256;
* the EnMAP branch maps ``[B, 218, 64, 64]`` to a full-resolution ``ori_feature``
  at 64x64 with 64 channels.
"""

import argparse

import torch

from _paths import add_to_path

add_to_path()

from lstd_mamba import LSTDSpectralEncoder, LSTDTimeEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test LSTD-Mamba branches")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--hyper-size", type=int, default=64)
    return parser.parse_args()


def check_time_branch(args: argparse.Namespace) -> None:
    print("== Sentinel-2 temporal branch (LST, LST, BTM) ==")
    model = LSTDTimeEncoder(
        in_channels=10,
        stage_channels=(64, 128, 256),
        depths=(1, 1, 1),
        btm_stage=2,
    ).to(args.device).eval()
    inputs = torch.randn(
        args.batch_size, args.frames, 10, args.size, args.size, device=args.device
    )
    with torch.inference_mode():
        outputs = model(inputs)

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    for index, feature in enumerate(outputs["stage_features"]):
        print(f"stage_features[{index}]: {tuple(feature.shape)}")

    widths = (64, 128, 256)
    sizes = [args.size // 4, args.size // 8, args.size // 16]
    for feature, channels, size in zip(
        outputs["stage_features"], widths, sizes, strict=True
    ):
        assert feature.shape == (args.batch_size, channels, size, size), (
            f"unexpected pooled stage shape {tuple(feature.shape)}"
        )
    # The pre-pooling features keep the acquisition axis.
    for feature, channels in zip(outputs["distill_temporal_features"], widths):
        assert feature.shape[1] == args.frames, "temporal axis must be preserved"
        assert feature.shape[2] == channels


def check_spectral_branch(args: argparse.Namespace) -> None:
    print("== EnMAP spectral branch (local projection + BSM) ==")
    model = LSTDSpectralEncoder(
        in_bands=218,
        spectral_channels=32,
        out_channels=64,
        depth=2,
        spatial_pool=4,
    ).to(args.device).eval()
    inputs = torch.randn(
        args.batch_size, 218, args.hyper_size, args.hyper_size, device=args.device
    )
    with torch.inference_mode():
        outputs = model(inputs)

    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    print(f"ori_feature: {tuple(outputs['ori_feature'].shape)}")
    assert outputs["ori_feature"].shape == (
        args.batch_size,
        64,
        args.hyper_size,
        args.hyper_size,
    ), "the EnMAP feature must stay at full resolution"


def main() -> None:
    args = parse_args()
    check_time_branch(args)
    check_spectral_branch(args)
    print("smoke test passed")


if __name__ == "__main__":
    main()

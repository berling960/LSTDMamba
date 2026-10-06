"""Build an LSTD-Mamba model and report its parameter budget.

Usage:
    python tools/check_model.py [CONFIG] [--build-only] [--device cuda]

With no CONFIG argument the bundled ``configs/lstd_mamba.py`` is used. Because
the reference network is 1.98M parameters, ``--build-only`` is the quickest way
to confirm that a configuration matches the paper's budget.
"""

import argparse
from pathlib import Path

import torch
from mmengine.config import Config

from _paths import PROJECT_ROOT, require_h2crop_root

require_h2crop_root()

from mmseg.registry import MODELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and check an LSTD-Mamba model"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(PROJECT_ROOT / "configs" / "lstd_mamba.py"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--hyper-size", type=int, default=64)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    args = parse_args()
    config = Config.fromfile(args.config)
    model = MODELS.build(config.model)

    components = {
        "s2_encoder": model.s2_encoder,
        "enmap_encoder": model.enmap_encoder,
        "neck": model.neck,
        "head": model.head,
    }
    for name, module in components.items():
        if module is None:
            continue
        print(f"{name}: {count_parameters(module) / 1e6:.3f} M")
    total = count_parameters(model)
    print(f"total: {total / 1e6:.3f} M")

    if args.build_only:
        return

    model = model.to(args.device).eval()
    inputs = {
        "S2": torch.randn(
            args.batch_size,
            args.frames,
            10,
            args.size,
            args.size,
            device=args.device,
        )
    }
    if config.get("with_enmap", True):
        inputs["EnMAP"] = torch.randn(
            args.batch_size,
            218,
            args.hyper_size,
            args.hyper_size,
            device=args.device,
        )
    priors = {}
    if config.get("with_priors", True):
        level_names = [
            level[0] if isinstance(level, (list, tuple)) else level
            for level in config.get("levels", [])
        ] or ["level1", "level2", "level3", "level4"]
        priors = {
            name: torch.zeros(
                args.batch_size,
                args.size,
                args.size,
                dtype=torch.long,
                device=args.device,
            )
            for name in level_names
        }
    with torch.inference_mode():
        outputs = model(inputs, priors, mode="tensor")[0]
    for level, logits in outputs.items():
        print(f"{level}: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

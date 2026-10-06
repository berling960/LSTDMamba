"""Evaluate an LSTD-Mamba checkpoint through the H2Crop test engine.

Usage:
    python tools/test.py CONFIG CHECKPOINT [H2Crop test options]

Each reported test result should use the checkpoint with the highest validation
mFscore, following the paper's protocol.
"""

import os
import runpy
import sys
from pathlib import Path

from _paths import PROJECT_ROOT, require_h2crop_root


def main() -> None:
    h2crop_root = require_h2crop_root()

    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python tools/test.py CONFIG CHECKPOINT [H2Crop test options]"
        )

    # PyTorch 2.6 defaults torch.load(..., weights_only=True). MMEngine
    # checkpoints may contain HistoryBuffer metadata, so loading a trusted
    # local training checkpoint needs the legacy full-checkpoint behavior.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    for index in (1, 2):
        path = Path(sys.argv[index])
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        sys.argv[index] = str(path)

    os.chdir(PROJECT_ROOT)
    runpy.run_path(str(h2crop_root / "test.py"), run_name="__main__")


if __name__ == "__main__":
    main()

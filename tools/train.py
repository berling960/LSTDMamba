"""Train LSTD-Mamba through the upstream H2Crop training engine.

Usage:
    python tools/train.py [CONFIG] [H2Crop train options]

``CONFIG`` defaults to ``configs/lstd_mamba.py`` when omitted. The paper's
protocol is 100 epochs on two GPUs with a batch size of 8 per GPU:

    torchrun --standalone --nproc_per_node=2 tools/train.py \\
        configs/lstd_mamba.py --launcher pytorch
"""

import os
import runpy
import sys
from pathlib import Path

from _paths import PROJECT_ROOT, require_h2crop_root


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lstd_mamba.py"


def main() -> None:
    h2crop_root = require_h2crop_root()

    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.argv.insert(1, str(DEFAULT_CONFIG))

    config_path = Path(sys.argv[1])
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    sys.argv[1] = str(config_path)

    os.chdir(PROJECT_ROOT)
    runpy.run_path(str(h2crop_root / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()

"""Shared path resolution for the LSTD-Mamba release.

The H2Crop dataset pipeline and training engine are distributed as a separate
upstream project, so its checkout location is resolved at runtime instead of
being hard-coded. Resolution order:

1. the ``H2CROP_ROOT`` environment variable, if set;
2. a sibling ``H2Crop-main`` directory next to the LSTD-Mamba checkout;
3. an already importable ``H2Crop`` package on ``sys.path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("H2CROP_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(PROJECT_ROOT / "H2Crop-main")
    candidates.append(PROJECT_ROOT.parent / "H2Crop-main")
    return candidates


def find_h2crop_root(required: bool = True) -> Path | None:
    """Return the directory that contains the H2Crop source tree."""
    for candidate in _candidate_roots():
        if (candidate / "H2Crop").is_dir() and (candidate / "mmseg").is_dir():
            return candidate
    if required:
        raise SystemExit(
            "H2Crop source tree not found.\n"
            "LSTD-Mamba reuses the upstream H2Crop dataset pipeline, hierarchy\n"
            "heads and mmseg registry. Clone it and point H2CROP_ROOT at the\n"
            "checkout, for example:\n\n"
            "  git clone <H2CROP_REPO_URL> H2Crop-main\n"
            "  export H2CROP_ROOT=$PWD/H2Crop-main\n"
        )
    return None


def add_to_path() -> Path | None:
    """Expose LSTD-Mamba and H2Crop imports to the current interpreter."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    root = find_h2crop_root(required=False)
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def require_h2crop_root() -> Path:
    """Like :func:`add_to_path` but fails loudly when H2Crop is missing."""
    root = add_to_path()
    if root is None:
        find_h2crop_root(required=True)
    return root

"""Where tests find the author's reference data files.

Set HOHYPEMB_SNAPSHOT_DIR to a folder holding the paper's data snapshot. A test that needs a
missing file is skipped (unittest.SkipTest), never failed.
"""
import os
import unittest
from pathlib import Path
from typing import Optional

_env = os.environ.get("HOHYPEMB_SNAPSHOT_DIR")
SNAPSHOT_DIR: Optional[Path] = Path(_env).expanduser() if _env else None


def snapshot_file(name: str) -> Optional[str]:
    """Path of a reference data file, or None when it is not available."""
    if SNAPSHOT_DIR is None:
        return None
    path = SNAPSHOT_DIR / name
    return str(path) if path.exists() else None


CORA_PATH = snapshot_file("cora-coauthorship.txt")


def require(path: Optional[str]) -> str:
    """Return ``path`` or skip the calling test when the data file is missing."""
    if path is None:
        raise unittest.SkipTest(
            "needs HOHYPEMB_SNAPSHOT_DIR pointing to the reference data files"
        )
    return path

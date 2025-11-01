from __future__ import annotations

import os
import sys


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def vendor_root() -> str:
    return os.path.join(repo_root(), "vendor", "rlm")


def bootstrap_paths() -> None:
    v = vendor_root()
    if v not in sys.path:
        sys.path.insert(0, v)


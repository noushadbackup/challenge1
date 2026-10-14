"""Locations of the (read-only) challenge package and of the feature cache."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PKG = Path(os.environ.get("ABC_ROOT", "/home/user/data_raw/abc2027_uwb_train_v1"))
DEFAULT_CACHE = Path(os.environ.get("UWBFALL_CACHE", "/home/user/data_raw/uwbfall_cache"))
RATE_HZ = 50.0
STEP_S = 0.2           # block length used by every model
FRAMES_PER_BLOCK = 10
SERIALS = (29171, 29444, 30722)

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only challenge package
if str(PKG / "src") not in sys.path:
    sys.path.insert(0, str(PKG / "src"))


def cache_dir(cfg: dict | None = None) -> Path:
    d = (cfg or {}).get("runtime", {}).get("cache_dir") if cfg else None
    p = Path(d) if d else DEFAULT_CACHE
    p.mkdir(parents=True, exist_ok=True)
    return p

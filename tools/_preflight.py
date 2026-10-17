"""Fail fast, before hours of training, when the Python environment cannot run the pipeline."""
from __future__ import annotations

import os
import sys


def check() -> None:
    missing = []
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "torch"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(f"preflight: {sys.executable} lacks {', '.join(missing)}. Activate the project environment first, "
                 f"e.g. `source ~/.venvs/uwbfall-gpu/bin/activate`, then rerun.")
    if os.environ.get("UWBFALL_DEVICE", "cpu").startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            sys.exit(f"preflight: DEVICE=cuda but torch {torch.__version__} in {sys.executable} sees no GPU.")
    import lightgbm
    import torch
    print(f"preflight: {sys.executable} | lightgbm {lightgbm.__version__} | torch {torch.__version__} | "
          f"device {os.environ.get('UWBFALL_DEVICE', 'cpu')}", flush=True)

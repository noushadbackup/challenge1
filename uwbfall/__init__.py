"""uwbfall: end-to-end ABC 2027 UWB fall-event pipeline (raw streams -> submission CSV).

    from uwbfall import fit, predict, load_config, split_sources
    bundle = fit(split_sources("train"), events_df, load_config(), "models/run1")
    df = predict(bundle, split_sources("val"))
"""
__version__ = "0.1.0"

import os as _os

# Fixed thread count before numpy / torch / OpenMP are loaded: reproducibility and no OpenMP spin-wait
# oversubscription on the shared 4-CPU machine. Override with UWBFALL_THREADS (or the usual env vars).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, _os.environ.get("UWBFALL_THREADS", "2"))

from .config import DEFAULT_CONFIG, load_config  # noqa: E402,F401
from .pipeline import fit, load_bundle, predict  # noqa: E402,F401
from .sources import H5Source, VirtualSource, dir_sources, parse_sources, sources_from_composition, split_sources  # noqa: E402,F401

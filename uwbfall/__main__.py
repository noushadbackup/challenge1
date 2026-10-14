import os

# fixed thread count before numpy / torch are imported (reproducibility, shared 4-CPU machine)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from .cli import main  # noqa: E402

raise SystemExit(main())

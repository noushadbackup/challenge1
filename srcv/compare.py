"""Apply the PLAN v2 promotion rule to two SR-CV prediction sets.

    python3 -m srcv.compare <A> <B> [--val-a 0.9646 --val-b 0.95] [--n-boot 5000] [--out result.json]

<A>/<B>: a results directory name under srcv/results/ (fold_k.csv or fold_k_*.csv files), or a comma list of such
names = seeds of one system (seed-averaged). B is the candidate, A the incumbent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import core as C
from . import scoring as S


def _arm(spec: str):
    arms = []
    for name in spec.split(","):
        d = Path(name) if Path(name).is_dir() else C.HERE / "results" / name
        files = {}
        for k in range(4):
            cand = sorted(d.glob(f"fold_{k}.csv")) or sorted(d.glob(f"fold_{k}_*.csv"))
            if len(cand) != 1:
                raise SystemExit(f"{d}: need exactly one fold_{k}[_*].csv, found {cand}")
            files[k] = cand[0]
        arms.append(files)
    return arms if len(arms) > 1 else arms[0]


def main() -> None:
    a = sys.argv[1:]
    get = lambda f, d=None: (a[a.index(f) + 1] if f in a else d)  # noqa: E731
    va, vb = get("--val-a"), get("--val-b")
    rep = S.promote(_arm(a[0]), _arm(a[1]), n_boot=int(get("--n-boot", 5000)),
                    val_A=float(va) if va else None, val_B=float(vb) if vb else None)
    print(json.dumps(rep, indent=1))
    if get("--out"):
        Path(get("--out")).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

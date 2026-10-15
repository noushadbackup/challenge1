"""Verify a model bundle against a shipped sha256 manifest, tolerating a legitimate refit.

    python3 -m uwbfall.tests.verify_bundle <bundle_dir> <shipped_dir>      (e.g. models/final)

Every file listed in <shipped_dir>/MANIFEST.sha256 must exist in <bundle_dir> with the same sha256, except
meta.json. meta.json must either match exactly, or differ from <shipped_dir>/meta.json only in fields that
predict never reads (timings, peak RSS, git rev, env). That is the case for a from-scratch `make train-final`
(audit, reports/round2/AUDIT.md F2). Predict reads only meta["postproc"]; the operating point is still checked.
Exit 0 = OK, 2 = mismatch.
"""
import hashlib
import json
import sys
from pathlib import Path

VOLATILE = {"timings", "peak_rss_gb", "git_rev", "env"}


def main(bundle: str, shipped: str) -> int:
    b, s = Path(bundle), Path(shipped)
    bad = 0
    for line in (s / "MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        h, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        p = b / name
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        if got == h:
            print(f"{name}: OK")
        elif name == "meta.json" and p.exists():
            a, r = json.loads((s / name).read_text()), json.loads(p.read_text())
            diff = sorted(k for k in set(a) | set(r) if a.get(k) != r.get(k))
            if set(diff) <= VOLATILE and a.get("postproc") == r.get("postproc"):
                print(f"{name}: OK (refit; differs only in non-predictive fields {diff})")
            else:
                print(f"{name}: FAILED (predictive fields differ: {sorted(set(diff) - VOLATILE)})")
                bad += 1
        else:
            print(f"{name}: FAILED ({got[:16]} != {h[:16]})")
            bad += 1
    print("BUNDLE CHECK:", "OK" if not bad else f"FAILED ({bad} files)")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

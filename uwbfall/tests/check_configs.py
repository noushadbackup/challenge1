"""Config invariants: reference.json == DEFAULT_CONFIG; the pinned configs differ from it only where intended.

    python3 -m uwbfall.tests.check_configs
"""
from pathlib import Path

from uwbfall.config import DEFAULT_CONFIG, check_implemented, load_config

C = Path(__file__).resolve().parents[1] / "configs"


def diff(a, b, p=""):
    out = []
    for k in a:
        if isinstance(a[k], dict) and k != "grid":
            out += diff(a[k], b[k], p + k + ".")
        elif a[k] != b[k]:
            out.append((p + k, a[k], b[k]))
    return out


ref = load_config(C / "reference.json")
assert ref == DEFAULT_CONFIG, diff(DEFAULT_CONFIG, ref)
assert load_config(C / "default.json") == DEFAULT_CONFIG
expect = {
    "C3G3.json": {"decode.one_event_per_clip": False, "clip.segclf.enabled": False, "clip.gate": "none"},
    "replayB_gate01.json": {"decode.one_event_per_clip": False, "clip.segclf.enabled": False, "clip.gate": "none",
                            "boundary.gate": 0.1},
    "integration_repro.json": {"decode.one_event_per_clip": False, "clip.segclf.enabled": False, "clip.gate": "none",
                               "boundary.gate": 0.1, "hgb.early_stopping": "auto"},
}
for f, e in expect.items():
    cfg = load_config(C / f)
    check_implemented(cfg)
    d = {k: v for k, _a, v in diff(DEFAULT_CONFIG, cfg)}
    assert d == e, (f, d)
    print(f"{f}: differs from the reference only in {d}")
check_implemented(ref)
print("configs OK")

"""Shared paths, loaders and plotting style for the analysis scripts."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PKG = Path(os.environ.get("ABC_ROOT", "/home/user/data_raw/abc2027_uwb_train_v1"))
FEATS = Path(os.environ.get("ABC_FEATS", "/home/user/data_raw/feats"))
REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RATE = 50.0
RADARS = ("R29171", "R29444", "R30722")

# Reference categorical palette (fixed order) and text inks.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.size": 9,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    }
)


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("figure", name)


@lru_cache(maxsize=1)
def events() -> pd.DataFrame:
    ev = pd.read_csv(PKG / "streams_events.csv")
    ev["split"] = ev.stream_id.str.rsplit("_", n=1).str[0]
    ev["dur"] = ev.t_end - ev.t_start
    ev["subject"] = ev.source_path.str.split("/").str[2]
    ev["flagged"] = (ev.quality == "low") | ev.needs_review.astype(str).str.lower().eq("true")
    return ev


@lru_cache(maxsize=1)
def composition() -> pd.DataFrame:
    co = pd.read_csv(PKG / "streams_composition.csv")
    co["split"] = co.stream_id.str.rsplit("_", n=1).str[0]
    co["dur"] = co.t_end - co.t_start
    co["group"] = np.where(co.kind == "adl", "ADL", np.where(co.kind == "fall", "Fall rec.", co.label))
    return co


def stream_ids() -> list[str]:
    return sorted(p.stem for p in FEATS.glob("*.npz"))


@lru_cache(maxsize=4)
def feats(sid: str) -> dict:
    with np.load(FEATS / f"{sid}.npz") as z:
        return {k: z[k] for k in z.files}


def frame_labels(sid: str, n: int, step: float) -> dict[str, np.ndarray]:
    """Per-block labels: segment kind/label of the block centre, fall mask, junction distance."""
    co = composition()[lambda d: d.stream_id == sid].sort_values("t_start")
    ev = events()[lambda d: d.stream_id == sid]
    t = (np.arange(n) + 0.5) * step
    kind = np.full(n, "", dtype=object)
    label = np.full(n, "", dtype=object)
    seg_start = np.zeros(n)
    for row in co.itertuples():
        m = (t >= row.t_start) & (t < row.t_end)
        kind[m], label[m], seg_start[m] = row.kind, row.label, row.t_start
    fall = np.zeros(n, bool)
    flagged = np.zeros(n, bool)
    for row in ev.itertuples():
        m = (t >= row.t_start) & (t < row.t_end)
        fall |= m
        flagged |= m & row.flagged
    edges = np.r_[co.t_start.values[1:]]
    jd = np.min(np.abs(t[:, None] - edges[None, :]), axis=1) if len(edges) else np.full(n, np.inf)
    return {"t": t, "kind": kind, "label": label, "fall": fall, "flagged": flagged, "junction_dist": jd, "seg_start": seg_start}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg)."""
    from scipy.stats import mannwhitneyu

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))

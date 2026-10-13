"""Reproduce every figure and headline number of reports/REPORT.md.

Usage: python analysis/make_figures.py   (after extract_features.py)
Writes reports/figures/*.png and reports/numbers.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from common import C, INK, INK2, PKG, RADARS, REPO, SEQ, auc, composition, events, feats, frame_labels, plt, save, stream_ids

NUM: dict = {}
ev, co = events(), composition()
DS = ["Dataset 1", "Dataset 2", "Dataset 3"]


def fig_composition():
    g = co.groupby(["split", "group"]).dur.sum().unstack(0).fillna(0) / 3600
    g = g.loc[["ADL", "Fall rec.", "Autre", "Empty"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    y = np.arange(len(g))
    for i, sp in enumerate(["train", "val"]):
        ax.barh(y + (i - 0.5) * 0.38, g[sp], height=0.36, color=C[i], label=sp)
        for yy, v in zip(y, g[sp]):
            ax.text(v + 0.05, yy + (i - 0.5) * 0.38, f"{v:.2f} h", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, ["ADL chunks", "Fall recordings", "'Autre' (other)", "Empty room"])
    ax.invert_yaxis()
    ax.set_xlabel("hours")
    ax.set_title("What the streams are made of")
    ax.legend()
    a = co[co.kind == "adl"].groupby("label").dur.sum().sort_values() / 60
    ax = axes[1]
    ax.barh(a.index, a.values, color=C[0], height=0.7)
    ax.set_xlabel("minutes (train + val)")
    ax.set_title("ADL time by activity")
    save(fig, "01_stream_composition")
    NUM["hours"] = g.round(3).to_dict()


def fig_timing():
    f = co[co.kind == "fall"].merge(ev[["event_id", "t_start", "t_end", "fall_type", "dataset"]],
                                    left_on="segment_id", right_on="event_id", suffixes=("_seg", "_ev"))
    f["pre"] = f.t_start_ev - f.t_start_seg
    f["post"] = f.t_end_seg - f.t_end_ev
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    bins = np.arange(0, 5.2, 0.2)
    for i, ft in enumerate(sorted(ev.fall_type.unique())):
        axes[0].hist(ev[ev.fall_type == ft].dur, bins=bins, histtype="step", lw=2, color=C[i], label=ft)
    axes[0].set_xlabel("annotated fall duration (s)")
    axes[0].set_ylabel("falls")
    axes[0].set_title("Fall duration: median 1.74 s")
    axes[0].legend()
    axes[1].hist(f.pre, bins=np.arange(0, 8.5, 0.5), color=C[0], rwidth=0.9)
    axes[1].set_xlabel("fall start − segment start (s)")
    axes[1].set_title("Fall starts 0–8 s after its junction")
    axes[2].hist(f.post, bins=np.arange(3, 10.5, 0.5), color=C[2], rwidth=0.9)
    axes[2].set_xlabel("segment end − fall end (s)")
    axes[2].set_title("…and ends 3–10 s before the next")
    save(fig, "02_fall_timing")
    NUM["fall_duration"] = ev.dur.describe().round(3).to_dict()
    NUM["fall_duration_by_type_median"] = ev.groupby("fall_type").dur.median().round(3).to_dict()
    NUM["pre_margin"] = f.pre.describe().round(3).to_dict()
    NUM["post_margin"] = f.post.describe().round(3).to_dict()
    NUM["fall_segment_len"] = f.dur.describe().round(3).to_dict()
    NUM["other_segment_len"] = co[co.kind == "other"].dur.describe().round(3).to_dict()
    NUM["adl_chunk_len"] = co[co.kind == "adl"].dur.describe().round(3).to_dict()
    gaps = ev.sort_values(["stream_id", "t_start"]).groupby("stream_id").t_start.diff().dropna()
    NUM["inter_fall_gap"] = gaps.describe().round(3).to_dict()
    NUM["falls_per_hour"] = round(len(ev) / (co.groupby("stream_id").t_end.max().sum() / 3600), 2)
    NUM["evaluable_falls"] = ev[~ev.flagged].groupby(["split", "dataset"]).size().unstack().fillna(0).astype(int).to_dict()
    NUM["flagged_rate"] = ev.groupby(["split", "dataset"]).flagged.mean().round(3).unstack().to_dict()


def junction_and_onset():
    """EMA vs frame-difference energy aligned on ADL->ADL junctions and on fall onsets."""
    W = (-200, 600)
    jun = {"ema": [], "diff": []}
    on = {k: {d: [] for d in DS} for k in ("ema", "diff")}
    adl_q = {"ema": [], "diff": []}
    for sid in stream_ids():
        F = feats(sid)
        T = F["e_mot"].shape[1]
        lab = frame_labels(sid, T, 0.02)
        c = co[co.stream_id == sid].sort_values("t_start")
        for key, arr in (("ema", F["e_mot"]), ("diff", F["e_diff"])):
            e = arr.astype(np.float64)
            ref = np.median(e[:, lab["kind"] == "adl"], axis=1, keepdims=True)
            en = np.convolve((e / ref).max(0), np.ones(10) / 10, "same")
            adl_q[key].append(en[(lab["kind"] == "adl") & (lab["junction_dist"] > 1)][::10])
            for a, b in zip(c.itertuples(), list(c.itertuples())[1:]):
                if a.kind == "adl" and b.kind == "adl":
                    s = int(round(b.t_start * 50))
                    if s + W[0] >= 0 and s + W[1] < T:
                        jun[key].append(en[s + W[0]: s + W[1]])
            for r in ev[ev.stream_id == sid].itertuples():
                s = int(round(r.t_start * 50))
                if s + W[0] >= 0 and s + W[1] < T:
                    on[key][r.dataset].append(en[s + W[0]: s + W[1]])
    tt = np.arange(W[0], W[1]) / 50
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for i, key in enumerate(("ema", "diff")):
        m = np.array(jun[key])
        med = np.median(m, 0)
        lo, hi = np.percentile(m, [25, 75], 0)
        axes[i].fill_between(tt, lo, hi, color=C[i], alpha=0.2, lw=0)
        axes[i].plot(tt, med, color=C[i])
        axes[i].axvline(0, color=INK, ls=":", lw=1)
        axes[i].set_yscale("log")
        axes[i].set_xlabel("time from an ADL→ADL junction (s)")
    axes[0].set_title("3 s EMA residual energy (official baselines)")
    axes[1].set_title("frame-difference energy (junction-immune)")
    axes[0].set_ylabel("energy / ADL median (median, IQR)")
    save(fig, "03_junction_artifact")
    NUM["junction_ema_peak_ratio_median"] = float(np.median(np.array(jun["ema"])[:, -W[0]: -W[0] + 25].max(1)))
    NUM["junction_diff_peak_ratio_median"] = float(np.median(np.array(jun["diff"])[:, -W[0] + 15: -W[0] + 40].max(1)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for i, key in enumerate(("ema", "diff")):
        q = np.percentile(np.concatenate(adl_q[key]), [50, 99])
        for j, d in enumerate(DS):
            m = np.array(on[key][d])
            axes[i].plot(tt, np.median(m, 0), color=C[j], label=d)
        axes[i].axhspan(q[0], q[1], color=INK2, alpha=0.08, lw=0, label="ADL median–p99")
        axes[i].axvline(0, color=INK, ls=":", lw=1)
        axes[i].set_yscale("log")
        axes[i].set_xlabel("time from annotated fall start (s)")
    axes[0].set_title("EMA energy around falls (junction ramp dominates)")
    axes[1].set_title("frame-difference energy around falls")
    axes[0].set_ylabel("energy / ADL median (median over falls)")
    axes[1].legend(loc="upper right")
    save(fig, "04_energy_around_falls")


def fig_range_time():
    W0, W1 = -4, 8
    nb = (W1 - W0) * 5
    acc = {d: np.zeros((3, nb, 184)) for d in DS}
    cnt = dict.fromkeys(DS, 0)
    for sid in stream_ids():
        mm = feats(sid)["mot_map"].astype(np.float32)
        n = mm.shape[1]
        lab = frame_labels(sid, n, 0.2)
        base = np.median(mm[:, lab["kind"] == "adl"], axis=1) + 1e-9
        z = mm / base[:, None, :]
        for r in ev[ev.stream_id == sid].itertuples():
            a = int(round((r.t_start + W0) * 5))
            if a >= 0 and a + nb <= n:
                acc[r.dataset] += z[:, a: a + nb]
                cnt[r.dataset] += 1
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True)
    peak = {}
    for i, d in enumerate(DS):
        m3 = acc[d] / cnt[d]
        for r in range(3):
            ax = axes[i, r]
            im = ax.imshow(np.log10(np.clip(m3[r].T, 1e-3, None)), aspect="auto", origin="lower", cmap=SEQ,
                           vmin=0, vmax=1.5, extent=[W0, W1, 0, 184], interpolation="nearest")
            ax.axvline(0, color=C[1], lw=1.2, ls="--")
            ax.grid(False)
            if i == 0:
                ax.set_title(RADARS[r])
            if r == 0:
                ax.set_ylabel(f"{d}\nrange bin (I/Q merged)")
            if i == 2:
                ax.set_xlabel("time from fall start (s)")
        win = m3[:, -W0 * 5: (-W0 + 2) * 5].mean(1)
        peak[d] = {RADARS[r]: {"bin": int(win[r].argmax()), "ratio": round(float(win[r].max()), 1)} for r in range(3)}
    fig.colorbar(im, ax=axes, shrink=0.6, label="log10( motion / ADL median motion, same bin )")
    fig.suptitle("Motion around fall onset (complex residual, 184 range bins)", fontweight="bold")
    save(fig, "05_fall_range_time_maps")
    NUM["fall_onset_peak_bins"] = peak


def static_and_fingerprint():
    rng = np.random.default_rng(0)
    rows, adl = [], []
    X, Y, S, K, IN = [], [], [], [], []
    for sid in stream_ids():
        a = feats(sid)["amp_1s"]
        n = a.shape[1]
        lab = frame_labels(sid, n, 1.0)
        for r in ev[ev.stream_id == sid].itertuples():
            b0, b1 = int(r.t_start) - 2, int(r.t_end) + 1
            if b0 >= 0 and b1 + 2 <= n:
                pre, post = a[:, b0], a[:, b1: b1 + 2].mean(1)
                rows.append((r.dataset, (np.linalg.norm(post - pre, axis=1) / np.linalg.norm(pre, axis=1)).max()))
        idx = np.flatnonzero(lab["kind"] == "adl")
        idx = idx[idx + 5 < n]
        for i in rng.choice(idx, 60):
            if lab["seg_start"][i] == lab["seg_start"][i + 5] and lab["kind"][i + 5] == "adl":
                pre, post = a[:, i], a[:, i + 3: i + 5].mean(1)
                adl.append((np.linalg.norm(post - pre, axis=1) / np.linalg.norm(pre, axis=1)).max())
        ok = (lab["junction_dist"] > 0.6) & (lab["kind"] != "")
        X.append(np.log10(a.transpose(1, 0, 2).reshape(n, -1) + 1e-9)[ok])
        Y += list(lab["kind"][ok] != "adl")
        K += list(lab["kind"][ok])
        S += [sid] * int(ok.sum())
        IN += list(lab["fall"][ok])
    d = pd.DataFrame(rows, columns=["ds", "sc"])
    adl = np.array(adl)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    bins = np.linspace(-2.5, 0.8, 40)
    axes[0].hist(np.log10(adl), bins=bins, density=True, color=INK2, alpha=0.35, label="ADL (same-segment, 4 s apart)")
    NUM["static_change_auc_vs_adl"] = {}
    for j, ds in enumerate(DS):
        v = d[d.ds == ds].sc.values
        axes[0].hist(np.log10(v), bins=bins, density=True, histtype="step", lw=2, color=C[j], label=ds)
        NUM["static_change_auc_vs_adl"][ds] = round(auc(v, adl), 3)
    axes[0].set_xlabel("log10 relative change of static profile (before vs after)")
    axes[0].set_title("Falls leave a lasting change in the static profile")
    axes[0].legend(fontsize=7)
    X = np.vstack(X)
    Y, K, S, IN = map(np.array, (Y, K, S, IN))
    tr = np.char.startswith(S.astype(str), "train")
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], Y[tr])
    p = m.predict_proba(X[~tr])[:, 1]
    kv, inv, yv = K[~tr], IN[~tr], Y[~tr]
    res = {
        "fall-corpus vs ADL (all)": roc_auc_score(yv, p),
        "fall-rec margins vs ADL": roc_auc_score(kv[(kv == "adl") | ((kv == "fall") & ~inv)] == "fall",
                                                 p[(kv == "adl") | ((kv == "fall") & ~inv)]),
        "fall-rec vs Autre/Empty": roc_auc_score(kv[kv != "adl"] == "fall", p[kv != "adl"]),
    }
    NUM["fingerprint_auc_val"] = {k: round(v, 4) for k, v in res.items()}
    axes[1].barh(list(res), list(res.values()), color=[C[0], C[0], C[3]], height=0.55)
    for i, v in enumerate(res.values()):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", color=INK)
    axes[1].axvline(0.5, color=INK2, ls=":", lw=1)
    axes[1].set_xlim(0.4, 1.05)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("validation AUC, one 1-s static amplitude block (train→val)")
    axes[1].set_title("Background alone reveals the source corpus")
    save(fig, "07_static_change_and_fingerprint")


def neighbour_activity():
    c = co.sort_values(["stream_id", "t_start"]).reset_index(drop=True)
    c["prev_label"] = c.groupby("stream_id").label.shift()
    c["prev_kind"] = c.groupby("stream_id").kind.shift()
    f = c[c.kind == "fall"]
    before = f[f.prev_kind == "adl"].prev_label.value_counts()
    share = c[c.kind == "adl"].groupby("label").size()
    t = pd.DataFrame({"before": before, "all": share}).fillna(0)
    t["before_pct"] = 100 * t.before / t.before.sum()
    t["expected_pct"] = 100 * t["all"] / t["all"].sum()
    t = t.sort_values("expected_pct")
    p = chi2_contingency(np.vstack([t.before.values, t["all"].values - t.before.values])).pvalue
    NUM["activity_before_fall_chi2_p"] = round(float(p), 3)
    NUM["neighbour_kind_before_fall"] = f.prev_kind.value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t.expected_pct, height=0.38, color=C[0], label="share of all ADL chunks")
    ax.barh(y + 0.2, t.before_pct, height=0.38, color=C[1], label="ADL chunk right before a fall")
    ax.set_yticks(y, t.index)
    ax.set_xlabel("% of chunks")
    ax.set_title(f"Activity before a fall is random (χ² p = {p:.2f})")
    ax.legend(loc="lower right")
    save(fig, "08_activity_before_fall")


def fig_prototype():
    s = json.load(open(REPO / "reports" / "prototype" / "summary.json"))
    names = ["Energy baseline", "Logistic baseline"] + [f"Prototype: {k}" for k in s]
    f1 = [[0.358, 0.338, 0.249, 0.060], [0.593, 0.167, 0.014, 0.000]] + [v["val_f1@0.1/0.3/0.5/0.7"] for v in s.values()]
    rank = [0.315, 0.258] + [v["val_rank_f1"] for v in s.values()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].barh(names, rank, color=[INK2, INK2, C[0], C[0], C[0]], height=0.55)
    for i, v in enumerate(rank):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", color=INK)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("ranking score = mean F1 @ θ 0.1/0.3/0.5 (val, --ignore-flagged)")
    axes[0].set_title("Official scorer on validation")
    th = [0.1, 0.3, 0.5, 0.7]
    for i, (n, v) in enumerate(zip(names, f1)):
        axes[1].plot(th, v, marker="o", ms=5, color=([INK2, "#a3a29c"] + C[:3])[i], label=n)
    axes[1].set_xticks(th)
    axes[1].set_xlabel("tIoU threshold θ")
    axes[1].set_ylabel("event F1")
    axes[1].set_title("Boundary precision is the remaining gap")
    axes[1].legend(fontsize=7)
    save(fig, "09_prototype_scores")


if __name__ == "__main__":
    fig_composition()
    fig_timing()
    junction_and_onset()
    fig_range_time()
    static_and_fingerprint()
    neighbour_activity()
    fig_prototype()
    json.dump(NUM, open(REPO / "reports" / "numbers.json", "w"), indent=1, default=str)
    print(json.dumps(NUM, indent=1, default=str))

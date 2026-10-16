"""Candidate tables + range-time crops for the image verifier (one stream at a time, no cross-stream info).

For every set S in fold0..fold3, official:
  train candidates (training streams of S's base bundle, never its test streams):
    src='final': the bundle's own OOF rows after replay + NMS (oof_final.csv)   -> the rows the verifier must judge
    src='low'  : extra detector candidates decoded from the bundle's OOF blend (oof.npz, not clip-gated) at a low
                 threshold (0.05, smooth 1, min_dur 0.2, gap 0.2), minus those overlapping a 'final' row (tIoU > 0.3)
                 -> more (hard) negatives
    src='rand' : N_RAND random 1.5 s windows per training stream, >= 3 s from any fall event and >= 2 s from any
                 other candidate (easy negatives; seeded)
  test candidates: the base system's predictions on S's test streams (work/STRICT/results/norep/fold_k.csv | val.csv)

Labels (training rows only are used for fitting/tuning): miou = max tIoU with any fall event of the stream (flagged
falls included, they are real falls); hit = miou >= 0.3. subject = subject of the composition segment that overlaps
the candidate most (grouping for subject-disjoint inner CV).

Crop: map (3 radars, 0.1 s bins, range bins 0..71) from maps.py; per stream/radar/range-bin median subtracted (stream's
own background); window = candidate centre +- 6 s (120 bins) x 72 range bins, float16.
Out: /home/user/data_raw/r3/verify/crops/<S>_{train,test}.npy and work/R3_verify/cands/<S>_{train,test}.csv
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

import srcv
from srcv import core as C
from uwbfall.decode import to_events, events_df
from uwbfall.clip import subject_of_path

RUNS = "/home/user/data_raw/srcv/runs/strict/norep"
BASE = "/home/user/challenge1/work/STRICT/results/norep"
MAPS = "/home/user/data_raw/r3/verify/maps"
CROPS = "/home/user/data_raw/r3/verify/crops"
CANDS = "/home/user/challenge1/work/R3_verify/cands"
HALF = 60  # bins (0.1 s) each side
LOW = (0.05, 1, 0.2, 0.2)
N_RAND = 25  # random background windows per training stream (easy negatives)
PKG = "/home/user/data_raw/abc2027_uwb_train_v1"


def tiou(a0, a1, b0, b1):
    inter = np.maximum(0, np.minimum(a1, b1) - np.maximum(a0, b0))
    union = np.maximum(a1, b1) - np.minimum(a0, b0)
    return inter / np.maximum(union, 1e-9)


def set_info(S):
    if S.startswith("fold"):
        k = int(S[4:])
        ev_tr, ev_te = C.events(k, "train"), C.events(k, "test")
        comp_tr, comp_te = C.composition(k, "train"), C.composition(k, "test")
        bundle = f"{RUNS}/fold_{k}"
        test_csv = f"{BASE}/fold_{k}.csv"
    else:
        ev_tr, ev_te = C.real_events("train"), C.real_events("val")
        comp = pd.read_csv(f"{PKG}/streams_composition.csv")
        comp_tr = comp[comp.stream_id.str.startswith("train_")].copy()
        comp_te = comp[comp.stream_id.str.startswith("val_")].copy()
        bundle = f"{RUNS}/official_train"
        test_csv = f"{BASE}/val.csv"
    return ev_tr, ev_te, comp_tr, comp_te, bundle, test_csv


def label(df, ev, comp):
    df = df.copy().reset_index(drop=True)
    miou = np.zeros(len(df))
    ign = np.zeros(len(df), bool)
    subj = []
    evg = {s: g for s, g in ev.groupby("stream_id")}
    cg = {s: g for s, g in comp.groupby("stream_id")}
    for i, r in df.iterrows():
        e = evg.get(r.stream_id)
        if e is not None and len(e):
            iou = tiou(r.t_start, r.t_end, e.t_start.astype(float).values, e.t_end.astype(float).values)
            miou[i] = iou.max()
            ig = e.ignore.astype(str).str.lower().values == "true"
            if ig.any():
                es, ee = e.t_start.astype(float).values[ig], e.t_end.astype(float).values[ig]
                ign[i] = bool(((r.t_start < ee + 2) & (r.t_end > es - 2)).any())
        c = cg[r.stream_id]
        cs, ce = c.t_start.astype(float).values, c.t_end.astype(float).values
        ov = np.minimum(ce, r.t_end) - np.maximum(cs, r.t_start)
        j = int(np.argmax(ov))
        subj.append(subject_of_path(c.path.values[j]))
    df["miou"] = miou.round(4)
    df["hit"] = (miou >= 0.3).astype(int)
    df["near_ignore"] = ign.astype(int)
    df["subject"] = subj
    return df


def low_cands(bundle, final):
    z = np.load(f"{bundle}/oof.npz", allow_pickle=True)
    sid, p = z["sid"], z["oof_blend"]
    evs = []
    for s in pd.unique(sid):
        evs += to_events(str(s), p[sid == s], *LOW)
    low = events_df(evs)
    keep = []
    fg = {s: g for s, g in final.groupby("stream_id")}
    for _, r in low.iterrows():
        f = fg.get(r.stream_id)
        keep.append(f is None or tiou(r.t_start, r.t_end, f.t_start.values, f.t_end.values).max() <= 0.3)
    return low[np.array(keep, bool)]


def rand_cands(ev, cand, durs, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for s, T in sorted(durs.items()):
        e = ev[ev.stream_id == s]
        es, ee = e.t_start.astype(float).values, e.t_end.astype(float).values
        c = cand[cand.stream_id == s]
        cc = 0.5 * (c.t_start.values + c.t_end.values)
        n = 0
        for _ in range(2000):
            if n >= N_RAND:
                break
            m = rng.uniform(8, T - 8)
            if ((m > es - 3) & (m < ee + 3)).any() or (np.abs(cc - m) < 2).any():
                continue
            rows.append((s, round(m - 0.75, 3), round(m + 0.75, 3), 0.0, round(m - 0.75, 3)))
            n += 1
    return pd.DataFrame(rows, columns=["stream_id", "t_start", "t_end", "score", "t_alarm"])


def crops(df):
    out = np.zeros((len(df), 3, 72, 2 * HALF), np.float16)
    for s, idx in df.groupby("stream_id").groups.items():
        m = np.load(f"{MAPS}/{s}.npy").astype(np.float32)  # (3, n, 72)
        m -= np.median(m, axis=1, keepdims=True)
        n = m.shape[1]
        for i in idx:
            c = int(round(0.5 * (df.t_start[i] + df.t_end[i]) * 10))
            a, b = c - HALF, c + HALF
            w = np.zeros((3, 2 * HALF, 72), np.float32)
            aa, bb = max(a, 0), min(b, n)
            w[:, aa - a: bb - a] = m[:, aa:bb]
            out[i] = np.transpose(w, (0, 2, 1)).astype(np.float16)
    return out


def main():
    os.makedirs(CROPS, exist_ok=True)
    os.makedirs(CANDS, exist_ok=True)
    for S in sys.argv[1:]:
        ev_tr, ev_te, comp_tr, comp_te, bundle, test_csv = set_info(S)
        final = pd.read_csv(f"{bundle}/oof_final.csv")
        low = low_cands(bundle, final)
        tr = pd.concat([final.assign(src="final"), low.assign(src="low")], ignore_index=True)
        durs = {s: np.load(f"{MAPS}/{s}.npy", mmap_mode="r").shape[1] / 10.0 for s in sorted(tr.stream_id.unique())}
        rnd = rand_cands(ev_tr, tr, durs, seed=sum(map(ord, S)))
        tr = pd.concat([tr, rnd.assign(src="rand")], ignore_index=True)
        tr = label(tr, ev_tr, comp_tr)
        te = label(pd.read_csv(test_csv).assign(src="test"), ev_te, comp_te)
        for part, df in (("train", tr), ("test", te)):
            df.to_csv(f"{CANDS}/{S}_{part}.csv", index=False)
            np.save(f"{CROPS}/{S}_{part}.npy", crops(df))
        f = tr[tr.src == "final"]
        l = tr[tr.src == "low"]
        print(f"{S}: train final {len(f)} (hit {f.hit.sum()}), low {len(l)} (hit {l.hit.sum()}), "
              f"rand {(tr.src == 'rand').sum()} (hit {tr[tr.src == 'rand'].hit.sum()}); "
              f"test {len(te)} (hit {te.hit.sum()}, near_ignore&miss {((te.hit == 0) & (te.near_ignore == 1)).sum()})",
              flush=True)


if __name__ == "__main__":
    main()

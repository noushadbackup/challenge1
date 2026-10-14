"""Step 2: replay the package annotator (abc2027.annotate.propose_fall) on several stream crops
around each candidate event and store the raw replay bounds per crop configuration.
No ground truth or composition csv is used here (junctions are detected from cached features).
Usage: python3 s2_replay.py <split> <candidates.csv> <out.csv>
"""
import sys, time
import h5py, numpy as np, pandas as pd
from scipy.ndimage import median_filter, maximum_filter

PKG = "/home/user/data_raw/abc2027_uwb_train_v1"
sys.path.insert(0, PKG + "/src")
from abc2027.annotate import RadarTrack, propose_fall  # noqa: E402

SERIALS = (29171, 29444, 30722)
FS = 50.0


def detected_junctions(sid, q=97):
    """Variance-normalised static-profile jump on 0.2 s blocks (X2/F4 definition). Returns times (s) and strength."""
    F = np.load(f"/home/user/data_raw/feats/{sid}.npz")
    p = F["prof_02"].astype(np.float32) / 1e3
    z = p[..., :184] + 1j * p[..., 184:]
    norm = np.median(np.linalg.norm(z, axis=2), axis=1, keepdims=True) + 1e-12
    d = np.zeros(z.shape[:2])
    d[:, 1:-1] = np.linalg.norm(z[:, 2:] - z[:, :-2], axis=2)
    j = (d / norm).max(0)
    jn = j / (median_filter(j, size=51, mode="nearest") + 1e-9)
    thr = np.percentile(jn, q)
    pk = np.flatnonzero((jn > thr) & (jn == maximum_filter(jn, size=11)))
    return (pk + 0.5) * 0.2, jn[pk]


def refine_junction(cir, t0, J, half=0.5):
    """Frame-level crossfade centre near block junction J using raw frames: argmax over t of
    sum_r ||mean(x[t:t+10]) - mean(x[t-10:t])||^2 / median level. cir starts at stream time t0."""
    i0 = int(round((J - half - t0) * FS)); i1 = int(round((J + half - t0) * FS))
    i0 = max(i0, 10); i1 = min(i1, cir.shape[1] - 10)
    if i1 <= i0:
        return J
    c = np.cumsum(np.pad(cir.astype(np.float64), ((0, 0), (1, 0), (0, 0))), axis=1)
    idx = np.arange(i0, i1)
    a = (c[:, idx + 10] - c[:, idx]) / 10
    b = (c[:, idx] - c[:, idx - 10]) / 10
    D = ((a - b) ** 2).sum(2)
    D = D / (np.median(D, axis=1, keepdims=True) + 1e-18)
    return t0 + idx[int(np.argmax(D.sum(0)))] / FS


def tracks_for(cir, dedup):
    tr = []
    for i, s in enumerate(SERIALS):
        x = cir[i].astype(np.float64)
        t = np.arange(len(x)) / FS
        if dedup:  # drop bit-exact sample-and-hold repeats, keep true frame times (annotator works on raw frames)
            keep = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
            x, t = x[keep], t[keep]
        d = np.diff(x, axis=0)
        tr.append(RadarTrack(serial=s, time_s=t, cir=x, motion=np.einsum("ij,ij->i", d, d)))
    return tr


# (name, use_junction, refine_junction, L, R, dedup). "d" suffix = motion on distinct frames (holds removed).
CONFIGS = [(f"loc{L:g}_{R:g}d", False, False, L, R, True) for L in (1.5, 2, 3) for R in (3, 5)]
CONFIGS += [(f"det{L:g}_{R:g}d", True, False, L, R, True) for L in (1.5, 2, 3, 5, 10) for R in (3, 5)]
CONFIGS += [(f"detr{L:g}_{R:g}d", True, True, L, R, True) for L in (2, 3) for R in (3, 5)]
CONFIGS += [(f"detq{L:g}_{R:g}d", "q95", False, L, R, True) for L in (2, 3, 5, 10) for R in (3, 5)]
CONFIGS += [(f"detqr{L:g}_{R:g}d", "q95", True, L, R, True) for L in (2, 3, 10) for R in (3, 5)]
CONFIGS += [("loc2_5", False, False, 2, 5, False), ("det2_5", True, False, 2, 5, False), ("det2_3", True, False, 2, 3, False)]

def main():
    split, cand, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    pred = pd.read_csv(cand)
    rows = []
    t_beg = time.time()
    for sid, P in pred.groupby("stream_id"):
        idx = int(sid.split("_")[1])
        h = h5py.File(f"{PKG}/streams/{split}/stream_{idx:02d}.h5", "r")
        Tn = h["cir"].shape[1]; T = Tn / FS
        J, Js = detected_junctions(sid, 97)
        J95, _ = detected_junctions(sid, 95)
        for r in P.itertuples():
            g = 1.0
            prev = J[J < r.t_start - g]; nxt = J[J > r.t_end + g]
            jp = prev[-1] if len(prev) else None
            jn = nxt[0] if len(nxt) else None
            # recall-oriented variant: lower threshold (q95) and a 0.2 s exclusion gate
            prev = J95[J95 < r.t_start - 0.2]; nxt = J95[J95 > r.t_end + 0.2]
            jp2 = prev[-1] if len(prev) else None
            jn2 = nxt[0] if len(nxt) else None
            lo = max(0.0, min(r.t_start - 10.5, (jp if jp is not None else r.t_start) - 1.0,
                              (jp2 if jp2 is not None else r.t_start) - 1.0))
            hi = min(T, max(r.t_end + 5.5, (jn if jn is not None else r.t_end) + 1.0))
            i_lo, i_hi = int(lo * FS), int(np.ceil(hi * FS))
            cir = h["cir"][:, i_lo:i_hi, :]
            t0 = i_lo / FS
            jpr = refine_junction(cir, t0, jp) if jp is not None else None
            jnr = refine_junction(cir, t0, jn) if jn is not None else None
            jpr2 = refine_junction(cir, t0, jp2) if jp2 is not None else None
            jnr2 = refine_junction(cir, t0, jn2) if jn2 is not None else None
            out = dict(stream_id=sid, t_start=r.t_start, t_end=r.t_end, score=r.score,
                       jprev=jp if jp is not None else np.nan, jnext=jn if jn is not None else np.nan,
                       jprev_ref=jpr if jpr is not None else np.nan, jnext_ref=jnr if jnr is not None else np.nan,
                       jprev2=jp2 if jp2 is not None else np.nan, jprev2_ref=jpr2 if jpr2 is not None else np.nan)
            for name, uj, rf, L, R, dd in CONFIGS:
                a, b = r.t_start - L, r.t_end + R
                if uj:
                    if uj == "q95":
                        pj, nj = (jpr2, jnr2) if rf else (jp2, jn2)
                    else:
                        pj, nj = (jpr, jnr) if rf else (jp, jn)
                    # crossfade occupies ~[J-0.1, J+0.1] around its centre; block peak is +-0.1 s
                    if pj is not None:
                        a = max(a, pj + (0.15 if rf else 0.2))
                    if nj is not None:
                        b = min(b, nj - (0.15 if rf else 0.2))
                a, b = max(a, lo, 0.0), min(b, hi, T)
                ia, ib = int(round((a - t0) * FS)), int(round((b - t0) * FS))
                if ib - ia < 150:  # < 3 s crop: not usable
                    out[f"{name}_s"] = out[f"{name}_e"] = np.nan
                    continue
                a = t0 + ia / FS
                p = propose_fall(tracks_for(cir[:, ia:ib, :], dd), (ib - ia) / FS)
                out[f"{name}_s"], out[f"{name}_e"] = a + p.t_start, a + p.t_end
                out[f"{name}_pk"] = a + p.t_peak
                out[f"{name}_rev"] = ";".join(p.reasons)
                out[f"{name}_lowq"] = p.quality == "low"
                out[f"{name}_anchor"] = SERIALS.index(p.anchor_serial)
                out[f"{name}_spread"] = p.onset_spread_s
                out[f"{name}_crop"] = (b - a)
            rows.append(out)
        h.close()
        print(sid, len(P), round(time.time() - t_beg, 1), flush=True)
    pd.DataFrame(rows).to_csv(outp, index=False)


if __name__ == "__main__":
    main()

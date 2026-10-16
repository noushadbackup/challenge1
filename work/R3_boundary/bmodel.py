"""Dense boundary model: per-0.02 s start / end log-probabilities from the 112-channel block features.

Input per 0.2 s block (all computed within ONE stream):
  * the 112 TCN channels of the base bundle (prototype 64 incl. the train-fitted PCA fingerprint + 48 bands),
    standardised with the bundle's tcn_norm (training statistics);
  * 30 'fine' channels: log10 of the per-radar motion energy e_mot at the 10 frames (50 Hz) inside the block,
    normalised by the stream median (no frame-repeat statistic: e_mot is the residual to a 3 s EMA);
  * optionally the detector's blended probability p (OOF on training streams, final members on test streams).
Output per block: 2 x 10 logits = start / end logit at each 0.02 s sub-position (fine grid of 50 Hz).

Training (TriDet-style distribution head with soft labels): for every GT fall inside a sampled window we take a
start window of +-W s around the GT start shifted by a random offset (simulating a detector error) and minimise
the cross-entropy between softmax(start logits over the window) and a Gaussian soft label (sigma) centred on the GT
start; same for the end. Only training-fold streams are used.
"""
from __future__ import annotations

from common import *  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from uwbfall.extract import get_features
from uwbfall.features import stream_blocks, design
from uwbfall.models import standardise

SUB = 10           # fine bins per block
DT = 0.02          # fine step (s)


def stream_inputs(src, B: dict, p: np.ndarray | None, use_p: bool = True, fine2: bool = False) -> np.ndarray:
    """(n, C) float32 input for one stream."""
    cfg = B["cfg"]
    F = get_features(src, cfg)
    blk = stream_blocks(F, cfg)
    xp, xb, _ = design(blk, B["pca"], cfg)
    X = np.hstack([xp.astype(np.float32), xb])
    X = standardise(X, B["mu"], B["sd"], cfg["tcn"]["clip_z"])
    n = X.shape[0]
    em = F["e_mot"].astype(np.float64)[:, : n * SUB]
    em = np.log10(em / (np.median(em, axis=1, keepdims=True) + 1e-12) + 1e-4)
    fine = em.reshape(3, n, SUB).transpose(1, 0, 2).reshape(n, 3 * SUB)
    fine = np.clip((fine - 0.0) / 1.5, -6, 6).astype(np.float32)
    parts = [X, fine]
    if fine2:   # frame-difference energy, 3-frame running max (no explicit repeat detection), per radar at 50 Hz
        from scipy.ndimage import maximum_filter1d
        ed = maximum_filter1d(F["e_diff"].astype(np.float64)[:, : n * SUB], 3, axis=1)
        ed = np.log10(ed / (np.median(ed, axis=1, keepdims=True) + 1e-12) + 1e-4)
        parts.append(np.clip(ed.reshape(3, n, SUB).transpose(1, 0, 2).reshape(n, 3 * SUB) / 1.5, -6, 6).astype(np.float32))
    if use_p:
        parts.append((p[:n] if p is not None else np.zeros(n, np.float32)).reshape(-1, 1).astype(np.float32) * 2 - 1)
    return np.hstack(parts)


class BNet(nn.Module):
    def __init__(self, cin: int, h: int = 64, dil=(1, 2, 4, 8, 16, 32), drop: float = 0.1, in_drop: float = 0.1):
        super().__init__()
        self.in_drop = nn.Dropout1d(in_drop)
        self.inp = nn.Conv1d(cin, h, 1)
        self.convs = nn.ModuleList([nn.Conv1d(h, h, 3, dilation=d, padding=d) for d in dil])
        self.mix = nn.ModuleList([nn.Conv1d(h, h, 1) for _ in dil])
        self.drop = nn.Dropout(drop)
        self.out = nn.Conv1d(h, 2 * SUB, 1)

    def forward(self, x):                      # x (b, C, n) -> (b, 2, n*SUB)
        x = self.inp(self.in_drop(x))
        for c, m in zip(self.convs, self.mix):
            x = x + self.drop(m(Fn.gelu(c(x))))
        o = self.out(x)                        # (b, 2*SUB, n)
        b, _, n = o.shape
        return o.view(b, 2, SUB, n).permute(0, 1, 3, 2).reshape(b, 2, n * SUB)


def _soft_ce(logits_w, center_idx, sigma_bins):
    """logits_w (L,), soft Gaussian target centred at float index."""
    L = logits_w.shape[0]
    j = torch.arange(L, dtype=torch.float32)
    tgt = torch.exp(-0.5 * ((j - center_idx) / sigma_bins) ** 2)
    tgt = tgt / tgt.sum()
    return -(tgt * torch.log_softmax(logits_w, 0)).sum()


def train_bnet(streams: list[tuple[np.ndarray, np.ndarray]], seed: int = 0, steps: int = 400, win: int = 600,
               batch: int = 8, W: float = 4.0, shift: float = 2.0, sigma: float = 0.06, lr: float = 2e-3,
               h: int = 64, noise: float = 0.05, log=None) -> BNet:
    """streams: list of (X (n,C), ev (k,2) [t_start,t_end] in s). Returns the trained model (CPU)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = streams[0][0].shape[1]
    model = BNet(cin, h)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    Xt = [torch.from_numpy(X.T.copy()) for X, _ in streams]
    # sample windows centred near falls (75 %) or uniformly (25 %)
    allev = [(i, e) for i, (_, ev) in enumerate(streams) for e in ev]
    wb = int(W / DT)
    sb = sigma / DT
    run = 0.0
    for st in range(steps):
        model.train()
        xs, metas = [], []
        for _ in range(batch):
            i, e = allev[rng.integers(len(allev))]
            n = Xt[i].shape[1]
            c = int(e[0] / 0.2) + int(rng.integers(-win // 2 + 40, win // 2 - 40))
            a = int(np.clip(c - win // 2, 0, max(0, n - win)))
            xs.append(Xt[i][:, a:a + win])
            metas.append((i, a))
        L = min(x.shape[1] for x in xs)
        xb = torch.stack([x[:, :L] for x in xs])
        xb = xb + noise * torch.randn_like(xb)
        out = model(xb)                         # (b, 2, L*SUB)
        loss, cnt = 0.0, 0
        for bi, (i, a) in enumerate(metas):
            t0 = a * 0.2
            for e in streams[i][1]:
                for k in (0, 1):
                    g = (e[k] - t0) / DT       # GT index on the window's fine grid
                    if g < wb + 1 or g > L * SUB - wb - 2:
                        continue
                    ctr = g + rng.uniform(-shift, shift) / DT
                    lo = int(round(ctr - wb)); hi = int(round(ctr + wb))
                    lo = max(lo, 0); hi = min(hi, L * SUB)
                    if not (lo <= g < hi):
                        continue
                    loss = loss + _soft_ce(out[bi, k, lo:hi], torch.tensor(g - lo, dtype=torch.float32), sb)
                    cnt += 1
        if cnt == 0:
            continue
        loss = loss / cnt
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.95 * run + 0.05 * float(loss) if st else float(loss)
        if log and (st + 1) % 100 == 0:
            log(f"  step {st+1}/{steps} loss {run:.3f}")
    return model.eval()


@torch.no_grad()
def predict_logits(model: BNet, X: np.ndarray) -> np.ndarray:
    """(2, n*SUB) float32 logits for one stream (time of fine bin j = j*DT + DT/2)."""
    model.eval()
    x = torch.from_numpy(X.T.copy()).unsqueeze(0)
    return model(x)[0].numpy().astype(np.float32)


def window_logp(lg: np.ndarray, center: float, W: float = 4.0):
    """log-softmax of a logit track over [center-W, center+W]: returns (times, logp)."""
    n = lg.shape[0]
    lo = max(0, int(round((center - W) / DT))); hi = min(n, int(round((center + W) / DT)))
    if hi - lo < 2:
        lo, hi = max(0, min(lo, n - 2)), min(n, max(hi, 2))
    w = lg[lo:hi].astype(np.float64)
    lp = w - (np.log(np.sum(np.exp(w - w.max()))) + w.max())
    return (np.arange(lo, hi) + 0.5) * DT, lp

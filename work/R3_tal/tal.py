"""Light anchor-free temporal action localisation (ActionFormer / TriDet style, conv backbone) for fall events
over the 112-channel 0.2 s block features of a whole stream.

Model
  stem: 2 x [conv3 -> channel LayerNorm -> ReLU] (112 -> H)
  level 0 (stride 1): ConvNeXt-1D residual blocks (depthwise conv k=7, dilations 1,2,4,8 + pointwise MLP)
  levels 1..L-1 (stride 2^l): maxpool(2) + 2 residual blocks (dilations 1,2)
  heads shared over levels (ActionFormer): cls (2 conv + conv -> 1 logit, prior 0.01), reg (2 conv + conv -> 2,
  ReLU(scale_l * x), distances to start / end in stride units); auxiliary per-block fall head on level 0.
Targets (ActionFormer): point (i + 0.5) * stride, positive if inside a fall, within 1.5 * stride of its centre
  and max(d_start, d_end) in the level's regression range; shortest fall wins.
Loss: sigmoid focal loss (EMA-normalised by #positives) + DIoU regression loss on positives + 0.5 * BCE aux head.
Decode: per level, points with p >= 0.01 -> segments; Gaussian soft-NMS (sigma 0.5) per stream; top 300.

Everything is computed within one stream (training windows / inference per stream); nothing is compared across
streams.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STEP = 0.2


class CLN(nn.Module):
    """LayerNorm over channels of (B, C, T)."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, h, k=7, d=1, drop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(h, h, k, padding=d * (k // 2), dilation=d, groups=h)
        self.ln = CLN(h)
        self.pw1 = nn.Conv1d(h, 2 * h, 1)
        self.pw2 = nn.Conv1d(2 * h, h, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.pw2(F.gelu(self.pw1(self.ln(self.dw(x))))))


class Head(nn.Module):
    def __init__(self, h, nout, prior=None):
        super().__init__()
        self.c1, self.n1 = nn.Conv1d(h, h, 3, padding=1), CLN(h)
        self.c2, self.n2 = nn.Conv1d(h, h, 3, padding=1), CLN(h)
        self.out = nn.Conv1d(h, nout, 3, padding=1)
        if prior is not None:
            nn.init.constant_(self.out.bias, -math.log((1 - prior) / prior))

    def forward(self, x):
        x = F.relu(self.n1(self.c1(x)))
        x = F.relu(self.n2(self.c2(x)))
        return self.out(x)


class TALNet(nn.Module):
    def __init__(self, cin=112, h=64, levels=4, in_drop=0.1, drop=0.1):
        super().__init__()
        self.levels = levels
        self.in_drop = nn.Dropout1d(in_drop)
        self.stem = nn.Sequential(nn.Conv1d(cin, h, 3, padding=1), CLN(h), nn.ReLU(),
                                  nn.Conv1d(h, h, 3, padding=1), CLN(h), nn.ReLU())
        self.l0 = nn.Sequential(*[Block(h, 7, d, drop) for d in (1, 2, 4, 8)])
        self.down = nn.ModuleList([nn.Sequential(Block(h, 7, 1, drop), Block(h, 7, 2, drop))
                                   for _ in range(levels - 1)])
        self.norms = nn.ModuleList([CLN(h) for _ in range(levels)])
        self.cls = Head(h, 1, prior=0.01)
        self.reg = Head(h, 2)
        self.scales = nn.Parameter(torch.ones(levels))
        self.aux = nn.Conv1d(h, 1, 1)

    def forward(self, x):
        x = self.l0(self.stem(self.in_drop(x)))
        feats = [x]
        for d in self.down:
            x = d(F.max_pool1d(x, 2, 2))
            feats.append(x)
        cls, reg = [], []
        for l, f in enumerate(feats):
            f = self.norms[l](f)
            cls.append(self.cls(f)[:, 0])
            reg.append(F.relu(self.scales[l] * self.reg(f)))
        return cls, reg, self.aux(feats[0])[:, 0]


# ----------------------------------------------------------------------------------------------- targets
RANGES = [(0, 4), (4, 8), (8, 16), (16, 1e9)]   # max(d_start, d_end) in blocks, per level
CENTER_R = 1.5


def targets(segs_blk: np.ndarray, T: int, levels: int):
    """segs_blk: (n, 2) fall [start, end) in block units (continuous) within the window [0, T).
    Returns per level: cls target (T_l,), reg target (T_l, 2) in stride units, positive mask."""
    out = []
    for l in range(levels):
        s = 2 ** l
        Tl = T // s
        p = (np.arange(Tl) + 0.5) * s
        cls = np.zeros(Tl, np.float32)
        reg = np.zeros((Tl, 2), np.float32)
        if len(segs_blk):
            a, b = segs_blk[:, 0][None], segs_blk[:, 1][None]
            ds, de = p[:, None] - a, b - p[:, None]
            inside = (ds > 0) & (de > 0)
            c = 0.5 * (a + b)
            r = CENTER_R * s
            near = (p[:, None] >= np.maximum(a, c - r)) & (p[:, None] <= np.minimum(b, c + r))
            mx = np.maximum(ds, de)
            lo, hi = RANGES[l] if l < len(RANGES) else RANGES[-1]
            ok = inside & near & (mx >= lo) & (mx < hi)
            ln = np.where(ok, (b - a), np.inf)
            j = ln.argmin(1)
            pos = np.isfinite(ln.min(1))
            cls[pos] = 1
            reg[pos, 0] = ds[pos, j[pos]] / s
            reg[pos, 1] = de[pos, j[pos]] / s
        out.append((cls, reg))
    return out


def focal(logits, t, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    loss = ce * (1 - pt) ** gamma
    return ((alpha * t + (1 - alpha) * (1 - t)) * loss).sum()


def diou(pred, tgt):
    """pred, tgt: (n, 2) distances (start, end) from the same point -> sum of DIoU losses."""
    ps, pe = pred[:, 0], pred[:, 1]
    ts, te = tgt[:, 0], tgt[:, 1]
    inter = torch.min(ps, ts) + torch.min(pe, te)
    union = ps + pe + ts + te - inter
    iou = inter / union.clamp(min=1e-6)
    enc = torch.max(ps, ts) + torch.max(pe, te)
    rho = 0.5 * ((pe - ps) - (te - ts))
    return (1 - iou + (rho / enc.clamp(min=1e-6)) ** 2).sum()


# ----------------------------------------------------------------------------------------------- training
def train(streams: list[dict], seed: int, steps: int = 1000, win: int = 512, batch: int = 8, h: int = 48,
          lr: float = 1e-3, wd: float = 0.05, noise: float = 0.05, aux_w: float = 0.5, log=print) -> TALNet:
    """streams: [{"X": (n, C) float, "ev": (m, 3) [t_start, t_end, flagged] seconds, "y": (n,)}]"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = streams[0]["X"].shape[1]
    model = TALNet(cin, h)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    lens = np.array([len(s["X"]) for s in streams], float)
    Xt = [torch.from_numpy(np.ascontiguousarray(s["X"].astype(np.float32).T)) for s in streams]
    ema_pos = 100.0
    t0 = time.time()
    model.train()
    for it in range(steps):
        idx = rng.choice(len(streams), batch, p=lens / lens.sum())
        xb, tg, yb = [], [[] for _ in range(model.levels)], []
        for i in idx:
            n = len(streams[i]["X"])
            a = int(rng.integers(0, n - win))
            xb.append(Xt[i][:, a:a + win])
            ev = streams[i]["ev"]
            sb = np.c_[ev[:, 0] / STEP - a, ev[:, 1] / STEP - a]
            sb = sb[(sb[:, 1] > 0) & (sb[:, 0] < win)]
            for l, (c, r) in enumerate(targets(sb, win, model.levels)):
                tg[l].append((c, r))
            yb.append(streams[i]["y"][a:a + win])
        xb = torch.stack(xb)
        xb = xb + noise * torch.randn_like(xb)
        cls, reg, aux = model(xb)
        lc, lr_, npos = 0.0, 0.0, 0
        for l in range(model.levels):
            ct = torch.from_numpy(np.stack([c for c, _ in tg[l]]))
            rt = torch.from_numpy(np.stack([r for _, r in tg[l]]))
            lc = lc + focal(cls[l], ct)
            m = ct > 0
            npos += int(m.sum())
            if m.any():
                lr_ = lr_ + diou(reg[l].permute(0, 2, 1)[m], rt[m])
        ema_pos = 0.9 * ema_pos + 0.1 * max(npos, 1)
        la = F.binary_cross_entropy_with_logits(aux, torch.from_numpy(np.stack(yb).astype(np.float32)))
        loss = (lc + lr_) / ema_pos + aux_w * la
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % 200 == 0:
            log(f"  step {it + 1}/{steps} loss {loss.item():.4f} (cls {float(lc) / ema_pos:.3f} reg "
                f"{float(lr_) / ema_pos:.3f} aux {la.item():.3f}) {time.time() - t0:.0f}s")
    model.eval()
    return model


# ----------------------------------------------------------------------------------------------- inference
@torch.no_grad()
def infer(model: TALNet, X: np.ndarray):
    """One stream -> (segments DataFrame-like array (k, 3) [t_start, t_end, score] before NMS, aux prob (n,),
    point-score per block (n,) = max over levels of cls prob of the cell covering the block)."""
    model.eval()
    n = len(X)
    m = 2 ** (model.levels - 1)
    T = int(math.ceil(n / m) * m)
    x = np.zeros((T, X.shape[1]), np.float32)
    x[:n] = X
    cls, reg, aux = model(torch.from_numpy(x.T.copy()).unsqueeze(0))
    segs, pmax = [], np.zeros(n, np.float32)
    for l in range(model.levels):
        s = 2 ** l
        p = torch.sigmoid(cls[l][0]).numpy()
        r = reg[l][0].numpy()
        pt = (np.arange(len(p)) + 0.5) * s
        valid = pt < n
        pb = np.repeat(p, s)[:n]
        pmax = np.maximum(pmax, pb)
        k = np.flatnonzero((p >= 0.01) & valid)
        if len(k):
            st = (pt[k] - r[0, k] * s) * STEP
            en = (pt[k] + r[1, k] * s) * STEP
            segs.append(np.c_[st, en, p[k]])
    segs = np.vstack(segs) if segs else np.zeros((0, 3))
    return segs, torch.sigmoid(aux[0]).numpy()[:n], pmax


def tiou_1n(a, B):
    inter = np.clip(np.minimum(a[1], B[:, 1]) - np.maximum(a[0], B[:, 0]), 0, None)
    union = (a[1] - a[0]) + (B[:, 1] - B[:, 0]) - inter
    return inter / np.maximum(union, 1e-9)


def soft_nms(segs: np.ndarray, sigma: float = 0.5, min_score: float = 0.001, max_n: int = 300) -> np.ndarray:
    S = segs.copy()
    keep = []
    while len(S) and len(keep) < max_n:
        i = S[:, 2].argmax()
        top = S[i].copy()
        if top[2] < min_score:
            break
        keep.append(top)
        S = np.delete(S, i, 0)
        if len(S):
            S[:, 2] *= np.exp(-(tiou_1n(top, S) ** 2) / sigma)
    return np.array(keep) if keep else np.zeros((0, 3))


def stream_outputs(model, X, dur):
    segs, aux, pmax = infer(model, X)
    segs[:, 0] = np.clip(segs[:, 0], 0, dur)
    segs[:, 1] = np.clip(segs[:, 1], 0, dur)
    segs = segs[segs[:, 1] - segs[:, 0] > 0.05]
    return soft_nms(segs), aux, pmax


def coverage(segs: np.ndarray, n: int) -> np.ndarray:
    """Per-block max score of the (soft-NMS) segments covering the block centre."""
    p = np.zeros(n, np.float32)
    t = (np.arange(n) + 0.5) * STEP
    for a, b, s in segs:
        i0, i1 = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
        p[i0:i1] = np.maximum(p[i0:i1], s)
    return p


# ----------------------------------------------------------------------------------------------- data
FEATS = "/home/user/data_raw/r3/tal/feats"


def load(name: str):
    z = np.load(f"{FEATS}/{name}.npz")
    meta = json.loads(str(z["meta"]))
    tr = {}
    for sid in meta["train"]:
        tr[sid] = {"X": z[f"X__{sid}"], "y": z[f"y__{sid}"], "fold": z[f"fold__{sid}"],
                   "ev": z[f"ev__{sid}"], "dur": meta["dur"][sid],
                   **{k: z[f"{k}__{sid}"] for k in ("oof_hgb", "oof_tcn", "oof_blend")}}
    te = {sid: {"X": z[f"X__{sid}"], "dur": meta["dur"][sid]} for sid in meta["test"]}
    return tr, te

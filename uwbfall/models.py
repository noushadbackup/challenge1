"""Frame-wise fall-probability models: prototype HGB and the E4 dilated TCN (bag of seeds).

TCN code is a port of work/E4_deep_model/tcn.py (same modules, sampler, RNG call order, loss), so that the
same seeds give the same weights with the same thread count.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier


# ----------------------------------------------------------------------------- HGB
def hgb_params(cfg: dict) -> dict:
    h = cfg["hgb"]
    es = h["early_stopping"]
    return dict(max_iter=h["max_iter"], learning_rate=h["learning_rate"], max_leaf_nodes=h["max_leaf_nodes"],
                l2_regularization=h["l2_regularization"], random_state=h["random_state"], early_stopping=es)


def fit_hgb(X: np.ndarray, y: np.ndarray, cfg: dict, w: np.ndarray | None = None) -> HistGradientBoostingClassifier:
    """w: optional per-block loss weights (labels.ignore_clips); rows with w == 0 are dropped, which is the same
    as zero sample weight for the loss and keeps the binning on the remaining rows."""
    if w is None:
        return HistGradientBoostingClassifier(**hgb_params(cfg)).fit(X, y)
    k = w > 0
    return HistGradientBoostingClassifier(**hgb_params(cfg)).fit(X[k], y[k])


# ----------------------------------------------------------------------------- TCN
def device() -> torch.device:
    """TCN device from $UWBFALL_DEVICE: cpu (default, bit-exact with the shipped bundles), cuda, or auto
    (cuda when available). Kept out of the config so bundles and their checksums do not depend on it."""
    want = os.environ.get("UWBFALL_DEVICE", "cpu").strip().lower() or "cpu"
    if want not in ("cpu", "cuda", "auto"):
        raise ValueError(f"UWBFALL_DEVICE must be cpu, cuda or auto, not {want!r}")
    if want == "cpu" or (want == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("UWBFALL_DEVICE=cuda but torch.cuda.is_available() is False "
                           "(install a CUDA build of torch, e.g. --index-url https://download.pytorch.org/whl/cu128)")
    return torch.device("cuda")


def setup_torch(cfg: dict):
    """Fix torch threads, deterministic kernels, and cap every native thread pool (OpenMP/BLAS) to
    runtime.threads. Returns the threadpoolctl limiter (keep a reference while working)."""
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(int(cfg["runtime"]["threads"]))
    lim = threadpool_limits(int(cfg["runtime"]["threads"]))
    dev = device()
    if dev.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(cfg["runtime"]["deterministic"])
    if cfg["runtime"]["deterministic"]:
        # CPU: strict (bit-exact). CUDA: deterministic where kernels exist, warn instead of failing otherwise.
        torch.use_deterministic_algorithms(True, warn_only=dev.type == "cuda")
    return lim


class Stage(nn.Module):
    def __init__(self, cin: int, h: int, dil: list[int], drop: float):
        super().__init__()
        self.inp = nn.Conv1d(cin, h, 1)
        self.convs = nn.ModuleList([nn.Conv1d(h, h, 3, dilation=d, padding=d) for d in dil])
        self.mix = nn.ModuleList([nn.Conv1d(h, h, 1) for _ in dil])
        self.drop = nn.Dropout(drop)
        self.out = nn.Conv1d(h, 1, 1)

    def forward(self, x):
        x = self.inp(x)
        for c, m in zip(self.convs, self.mix):
            x = x + self.drop(m(F.gelu(c(x))))
        return self.out(x).squeeze(1)


class MSTCN(nn.Module):
    def __init__(self, cin: int, h: int, stages: int, drop: float, in_drop: float, dil=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        dil = list(dil)
        self.in_drop = nn.Dropout1d(in_drop)
        self.stages = nn.ModuleList([Stage(cin, h, dil, drop)] + [Stage(1, h // 2, dil[:6], drop) for _ in range(stages - 1)])

    def forward(self, x):
        outs = [self.stages[0](self.in_drop(x))]
        for s in self.stages[1:]:
            outs.append(s(torch.sigmoid(outs[-1]).unsqueeze(1)))
        return outs


def build_tcn(cin: int, cfg: dict) -> MSTCN:
    t = cfg["tcn"]
    return MSTCN(cin, t["hidden"], t["stages"], t["drop"], t["in_drop"], t["dilations"])


def seqs(X, y, sid, keep, w=None):
    """Per-stream (X^T, y) tensors; with w (per-block loss weights) the tuples carry a third weight tensor."""
    out = []
    for s in np.unique(sid[keep]):
        m = sid == s
        t = (torch.from_numpy(X[m].T.copy()), torch.from_numpy(y[m].astype(np.float32)))
        out.append(t if w is None else t + (torch.from_numpy(w[m].astype(np.float32)),))
    return out


def _loss(outs, yb):
    loss = 0.0
    for o in outs:
        loss = loss + F.binary_cross_entropy_with_logits(o, yb)
        lp = F.logsigmoid(o)
        loss = loss + 0.15 * torch.clamp((lp[:, 1:] - lp[:, :-1].detach()) ** 2, max=16).mean() if len(outs) > 1 else loss
    return loss


def _wloss(outs, yb, wb):
    """BCE with per-block weights (single-stage TCN; weights 0 = ignore zone). Mean over weighted blocks."""
    loss = 0.0
    for o in outs:
        b = F.binary_cross_entropy_with_logits(o, yb, reduction="none")
        loss = loss + (b * wb).sum() / wb.sum().clamp(min=1.0)
    return loss


def train_tcn(tr_seqs, cfg: dict, seed: int, epochs: int) -> MSTCN:
    t = cfg["tcn"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = tr_seqs[0][0].shape[0]
    dev = device()
    model = build_tcn(cin, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    lens = np.array([s[1].shape[0] for s in tr_seqs], float)
    for _ep in range(epochs):
        model.train()
        for _ in range(t["steps"]):
            idx = rng.choice(len(tr_seqs), t["batch"], p=lens / lens.sum())
            xb, yb, wb = [], [], []
            for i in idx:
                x, y = tr_seqs[i][:2]
                a = rng.integers(0, y.shape[0] - t["win"])
                xb.append(x[:, a : a + t["win"]])
                yb.append(y[a : a + t["win"]])
                if len(tr_seqs[i]) == 3:
                    wb.append(tr_seqs[i][2][a : a + t["win"]])
            xb = torch.stack(xb).to(dev)
            xb = xb + t["noise"] * torch.randn_like(xb)
            if wb:
                if len(model.stages) > 1:
                    raise NotImplementedError("loss weights are implemented for the single-stage TCN only")
                loss = _wloss(model(xb), torch.stack(yb).to(dev), torch.stack(wb).to(dev))
            else:
                loss = _loss(model(xb), torch.stack(yb).to(dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu()  # bundles always hold CPU weights


@torch.no_grad()
def predict_seq(model: MSTCN, X: np.ndarray) -> np.ndarray:
    """X: (n, C) standardised float32 for ONE stream -> (n,) float32 probabilities."""
    model.eval()
    dev = device()
    model.to(dev)
    x = torch.from_numpy(X.T.copy()).unsqueeze(0).to(dev)
    return torch.sigmoid(model(x)[-1][0]).cpu().numpy().astype(np.float32)


def standardise(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float) -> np.ndarray:
    return np.clip((X - mu) / sd, -clip, clip).astype(np.float32)

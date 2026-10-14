"""Frame-wise fall-probability models: prototype HGB and the E4 dilated TCN (bag of seeds).

TCN code is a port of work/E4_deep_model/tcn.py (same modules, sampler, RNG call order, loss), so that the
same seeds give the same weights with the same thread count.
"""
from __future__ import annotations

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


def fit_hgb(X: np.ndarray, y: np.ndarray, cfg: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**hgb_params(cfg)).fit(X, y)


# ----------------------------------------------------------------------------- TCN
def setup_torch(cfg: dict):
    """Fix torch threads, deterministic kernels, and cap every native thread pool (OpenMP/BLAS) to
    runtime.threads. Returns the threadpoolctl limiter (keep a reference while working)."""
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(int(cfg["runtime"]["threads"]))
    lim = threadpool_limits(int(cfg["runtime"]["threads"]))
    if cfg["runtime"]["deterministic"]:
        torch.use_deterministic_algorithms(True)
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


def seqs(X, y, sid, keep):
    out = []
    for s in np.unique(sid[keep]):
        m = sid == s
        out.append((torch.from_numpy(X[m].T.copy()), torch.from_numpy(y[m].astype(np.float32))))
    return out


def _loss(outs, yb):
    loss = 0.0
    for o in outs:
        loss = loss + F.binary_cross_entropy_with_logits(o, yb)
        lp = F.logsigmoid(o)
        loss = loss + 0.15 * torch.clamp((lp[:, 1:] - lp[:, :-1].detach()) ** 2, max=16).mean() if len(outs) > 1 else loss
    return loss


def train_tcn(tr_seqs, cfg: dict, seed: int, epochs: int) -> MSTCN:
    t = cfg["tcn"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = tr_seqs[0][0].shape[0]
    model = build_tcn(cin, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    lens = np.array([s[1].shape[0] for s in tr_seqs], float)
    for _ep in range(epochs):
        model.train()
        for _ in range(t["steps"]):
            idx = rng.choice(len(tr_seqs), t["batch"], p=lens / lens.sum())
            xb, yb = [], []
            for i in idx:
                x, y = tr_seqs[i]
                a = rng.integers(0, y.shape[0] - t["win"])
                xb.append(x[:, a : a + t["win"]])
                yb.append(y[a : a + t["win"]])
            xb = torch.stack(xb)
            xb = xb + t["noise"] * torch.randn_like(xb)
            loss = _loss(model(xb), torch.stack(yb))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict_seq(model: MSTCN, X: np.ndarray) -> np.ndarray:
    """X: (n, C) standardised float32 for ONE stream -> (n,) float32 probabilities."""
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(X.T.copy()).unsqueeze(0))[-1][0]).numpy().astype(np.float32)


def standardise(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, clip: float) -> np.ndarray:
    return np.clip((X - mu) / sd, -clip, clip).astype(np.float32)

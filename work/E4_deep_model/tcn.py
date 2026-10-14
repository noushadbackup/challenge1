"""Deep temporal model: dilated residual TCN (optionally 2-stage MS-TCN) over per-0.2 s block features.

  python3 tcn.py --tag <name> [--feats proto|all] [--stages 1|2] [--hidden 64] [--seeds 3] [--max_epochs 40]

Protocol (train streams only for every choice):
  1. 4-fold leave-streams-out CV (the prototype's GroupKFold folds). For every fold and seed, train for
     max_epochs and record held-out block BCE + held-out probabilities after every epoch.
  2. E* = epoch with the lowest mean held-out BCE across folds/seeds (one shared "early-stopping" epoch).
     OOF probabilities = seed-mean held-out probabilities at E*.
  3. Post-processing (prototype grid) tuned on OOF with the official evaluator code (in-process).
  4. Final model: all 13 train streams, E* epochs, same seeds; val probabilities = seed mean.
     Val scored once with the official evaluator CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lib import OUT, events_from_probs, load, official, summary, tune  # noqa: E402

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


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
    def __init__(self, cin: int, h: int, stages: int, drop: float, in_drop: float):
        super().__init__()
        dil = [1, 2, 4, 8, 16, 32, 64]  # receptive field 255 blocks = 51 s
        self.in_drop = nn.Dropout1d(in_drop)
        self.stages = nn.ModuleList([Stage(cin, h, dil, drop)] + [Stage(1, h // 2, dil[:6], drop) for _ in range(stages - 1)])

    def forward(self, x):
        outs = [self.stages[0](self.in_drop(x))]
        for s in self.stages[1:]:
            outs.append(s(torch.sigmoid(outs[-1]).unsqueeze(1)))
        return outs  # list of logits (B, T)


def make_X(D: dict, split: str, feats: str) -> np.ndarray:
    X = D[f"{split}_X_proto"].astype(np.float32)
    if feats == "all":
        X = np.hstack([X, D[f"{split}_X_band"]])
    return X


def seqs(X, y, sid, keep):
    out = []
    for s in np.unique(sid[keep]):
        m = sid == s
        out.append((torch.from_numpy(X[m].T.copy()), torch.from_numpy(y[m].astype(np.float32))))
    return out


def loss_fn(outs, yb):
    loss = 0.0
    for o in outs:
        loss = loss + F.binary_cross_entropy_with_logits(o, yb)
        lp = F.logsigmoid(o)
        loss = loss + 0.15 * torch.clamp((lp[:, 1:] - lp[:, :-1].detach()) ** 2, max=16).mean() if len(outs) > 1 else loss
    return loss


def train(tr_seqs, args, seed, epochs, eval_fn=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cin = tr_seqs[0][0].shape[0]
    model = MSTCN(cin, args.hidden, args.stages, args.drop, args.in_drop)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lens = np.array([s[1].shape[0] for s in tr_seqs], float)
    hist = []
    for ep in range(epochs):
        model.train()
        for _ in range(args.steps):
            idx = rng.choice(len(tr_seqs), args.batch, p=lens / lens.sum())
            xb, yb = [], []
            for i in idx:
                x, y = tr_seqs[i]
                a = rng.integers(0, y.shape[0] - args.win)
                xb.append(x[:, a : a + args.win])
                yb.append(y[a : a + args.win])
            xb = torch.stack(xb)
            xb = xb + args.noise * torch.randn_like(xb)
            loss = loss_fn(model(xb), torch.stack(yb))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if eval_fn is not None:
            hist.append(eval_fn(model))
    return model, hist


@torch.no_grad()
def predict(model, X, sid):
    model.eval()
    p = np.zeros(len(X), np.float32)
    for s in np.unique(sid):
        m = sid == s
        p[m] = torch.sigmoid(model(torch.from_numpy(X[m].T.copy()).unsqueeze(0))[-1][0]).numpy()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feats", default="proto", choices=["proto", "all"])
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--in_drop", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--win", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no_val", action="store_true", help="CV only, do not touch val")
    args = ap.parse_args()
    t0 = time.time()
    D = load()
    Xtr, Xva = make_X(D, "train", args.feats), make_X(D, "val", args.feats)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6  # label-free standardisation
    Xtr = np.clip((Xtr - mu) / sd, -6, 6).astype(np.float32)
    Xva = np.clip((Xva - mu) / sd, -6, 6).astype(np.float32)
    ytr, sid, fold = D["train_y"], D["train_sid"], D["train_fold"]

    # 1. CV with per-epoch held-out tracking
    E = args.max_epochs
    probs = np.zeros((args.seeds, E, len(ytr)), np.float32)
    bce = np.zeros((args.seeds, 4, E))
    for k in range(4):
        tr_seqs = seqs(Xtr, ytr, sid, fold != k)
        ho = fold == k
        for s in range(args.seeds):
            def ev(model, s=s, k=k):
                p = predict(model, Xtr[ho], sid[ho])
                pc = np.clip(p, 1e-6, 1 - 1e-6)
                yy = ytr[ho]
                ep = len([1 for _ in range(0)])  # placeholder
                return p, float(-(yy * np.log(pc) + (1 - yy) * np.log(1 - pc)).mean())
            _, hist = train(tr_seqs, args, seed=100 * s + k, epochs=E, eval_fn=ev)
            for e, (p, b) in enumerate(hist):
                probs[s, e, ho] = p
                bce[s, k, e] = b
        print(f"fold {k} done  bce/epoch(seed-mean) min={bce[:, k].mean(0).min():.4f} at {bce[:, k].mean(0).argmin() + 1}  t={time.time() - t0:.0f}s", flush=True)
    curve = bce.mean((0, 1))
    e_star = int(curve.argmin())
    oof = probs[:, e_star].mean(0)
    del probs
    sc, g = tune(sid, D["train_t"], oof)
    print(f"E*={e_star + 1}  oof_bce={curve[e_star]:.4f}  train OOF rank F1={sc:.4f} postproc={g}  t={time.time() - t0:.0f}s", flush=True)
    res = {"args": vars(args), "e_star": e_star + 1, "bce_curve": [round(float(c), 5) for c in curve],
           "train_oof_rank_f1": round(sc, 4), "postproc": g}
    np.savez(OUT / f"probs_{args.tag}.npz", oof=oof)
    if not args.no_val:
        # 4. final fit on all train streams, E* epochs, seed-mean
        all_seqs = seqs(Xtr, ytr, sid, np.ones(len(ytr), bool))
        pv = np.mean([predict(train(all_seqs, args, seed=1000 + s, epochs=e_star + 1)[0], Xva, D["val_sid"]) for s in range(args.seeds)], 0)
        np.savez(OUT / f"probs_{args.tag}.npz", oof=oof, val=pv)
        r = official(events_from_probs(D["val_sid"], D["val_t"], pv, g), f"val_{args.tag}")
        res["val"] = summary(r)
    res["wall_s"] = round(time.time() - t0)
    print(json.dumps({k: v for k, v in res.items() if k != "bce_curve"}), flush=True)
    json.dump(res, open(OUT / f"res_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Image verifier: ImageNet-pretrained ResNet-18 (torchvision weights resnet18-f37072fd.pth, re-implemented in plain
torch because torchvision is not installed) fine-tuned on CPU, 1 thread, on range-time crops (3 radars x 72 range bins x
80 time bins = candidate centre +- 4 s) to predict hit = tIoU >= 0.3 with a true fall.

usage: python cnn.py <set> <mode> [--seed s] [--epochs e] [--tag name] [--half 40|60] [--chans 012|012345]
  set : fold0..fold3 | official
  mode: inner -> subject-grouped 4-fold OOF on the set's TRAINING candidates (for tuning the cut)
        full  -> train on all training candidates, predict the set's test candidates (base predictions)
out : work/R3_verify/probs/<tag>/<set>_<mode>_s<seed>.csv (training or test candidate rows + p)
"""
import argparse
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

torch.set_num_threads(1)
CROPS = "/home/user/data_raw/r3/verify/crops2"
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = "/home/user/data_raw/r3/verify/resnet18-f37072fd.pth"
W = 40  # half window (bins of 0.1 s); set by --half
CH = [0, 1, 2]  # crop channels (crops2: D_0..2 = frame-diff energy, A_0..2 = amplitude); set by --chans
JIT = 5  # train-time centre jitter (bins)


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        cfg = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
        for i, (a, b, s) in enumerate(cfg, 1):
            setattr(self, f"layer{i}", nn.Sequential(Block(a, b, s), Block(b, b, 1)))
        self.fc = nn.Linear(512, 1000)

    def features(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.mean((2, 3))


class Net(nn.Module):
    def __init__(self, pretrained=True, nch=3):
        super().__init__()
        self.body = ResNet18()
        if pretrained:
            self.body.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        if nch != 3:  # tile the pretrained RGB filters over nch inputs (sum preserved)
            w = self.body.conv1.weight.data
            self.body.conv1 = nn.Conv2d(nch, 64, 7, 2, 3, bias=False)
            self.body.conv1.weight.data = torch.cat([w] * (nch // 3), 1) / (nch // 3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))

    def forward(self, x):
        # x: (B, 3, 72, 80) -> upsample to 96 x 112 so the 32x stride leaves a 3 x 4 map
        x = F.interpolate(x, size=(96, 32 * round(x.shape[3] * 1.4 / 32)), mode="bilinear", align_corners=False)
        return self.head(self.body.features(x)).squeeze(1)


def prep(X, centre_shift=None, rng=None, train=False):
    """X (n, 3, 72, 2*HALF) float16 -> (n, 3, 72, 2W) float32 tensor, scaled to ~unit range."""
    n, half = X.shape[0], X.shape[3] // 2
    if train:
        sh = rng.integers(-JIT, JIT + 1, n)
    else:
        sh = np.zeros(n, int)
    out = np.empty((n, len(CH), 72, 2 * W), np.float32)
    for i in range(n):
        a = half - W + sh[i]
        out[i] = X[i, CH, :, a: a + 2 * W]
    amp = [j for j, c in enumerate(CH) if c >= 3]
    out[:, amp] *= 4.0  # amplitude channels have ~1/4 of the frame-difference spread
    out = np.clip(out, -2, 5) / 1.5
    if train:  # per-radar gain jitter (+-0.3 dex) and small noise
        out += (rng.normal(0, 0.2, (n, len(CH), 1, 1)) + rng.normal(0, 0.05, out.shape)).astype(np.float32)
    return torch.from_numpy(out)


def fit_predict(Xtr, ytr, Xte, seed, epochs, log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(nch=len(CH))
    body = [p for p in net.body.parameters()]
    opt = torch.optim.AdamW([{"params": body, "lr": 3e-4}, {"params": net.head.parameters(), "lr": 1e-3}],
                            weight_decay=1e-4)
    bs = 32
    steps = epochs * int(np.ceil(len(ytr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=steps, pct_start=0.15)
    pw = torch.tensor(float((ytr == 0).sum()) / max(1, (ytr == 1).sum()))
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(ytr))
        tl, t0 = 0.0, time.time()
        for b in range(0, len(perm), bs):
            idx = perm[b: b + bs]
            if len(idx) < 2:
                continue
            xb = prep(Xtr[idx], rng=rng, train=True)
            yb = torch.from_numpy(ytr[idx].astype(np.float32))
            loss = F.binary_cross_entropy_with_logits(net(xb), yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(idx)
        log(f"  ep {ep} loss {tl / len(ytr):.4f} {time.time() - t0:.0f}s")
    net.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(Xte), 128):
            out.append(torch.sigmoid(net(prep(Xte[b: b + 128]))).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("mode", choices=["inner", "full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--tag", default="r18")
    ap.add_argument("--half", type=int, default=40)
    ap.add_argument("--chans", default="012")
    ap.add_argument("--no-rand", action="store_true", help="train on the hard candidates only (no random windows)")
    a = ap.parse_args()
    global W
    W = a.half
    CH[:] = [int(c) for c in a.chans]
    out_dir = f"{HERE}/probs/{a.tag}"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{a.set}_{a.mode}_s{a.seed}.csv"
    log = lambda m: print(m, flush=True)  # noqa: E731
    tr = pd.read_csv(f"{HERE}/cands/{a.set}_train.csv")
    Xtr = np.load(f"{CROPS}/{a.set}_train.npy")
    if a.no_rand:
        keep = (tr.src != "rand").values
        tr, Xtr = tr[keep].reset_index(drop=True), Xtr[keep]
    ytr = tr.hit.values.astype(int)
    t0 = time.time()
    if a.mode == "inner":
        p = np.full(len(tr), np.nan)
        for j, (i_tr, i_te) in enumerate(GroupKFold(4).split(Xtr, ytr, tr.subject.values)):
            log(f"{a.set} inner fold {j}: train {len(i_tr)} (pos {ytr[i_tr].sum()}), test {len(i_te)}")
            p[i_te] = fit_predict(Xtr[i_tr], ytr[i_tr], Xtr[i_te], a.seed * 100 + j, a.epochs, log)
        res = tr.assign(p=p)
    else:
        te = pd.read_csv(f"{HERE}/cands/{a.set}_test.csv")
        Xte = np.load(f"{CROPS}/{a.set}_test.npy")
        log(f"{a.set} full: train {len(tr)} (pos {ytr.sum()}), test {len(te)}")
        res = te.assign(p=fit_predict(Xtr, ytr, Xte, a.seed * 100 + 99, a.epochs, log))
    res.to_csv(out, index=False)
    log(f"done {out} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

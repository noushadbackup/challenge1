"""Seed variance of the final TCN on val: 5 single-seed models (all train streams, E*=8 epochs,
the 'tcn_all' config), each scored with the train-OOF-tuned post-processing. Nothing is tuned here."""
import json
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

from lib import OUT, events_from_probs, load, score_inproc  # noqa: E402
from tcn import make_X, predict, seqs, train  # noqa: E402

cfg = json.load(open(OUT / "res_tcn_all.json"))
args = SimpleNamespace(**cfg["args"])
g = tuple(cfg["postproc"])
D = load()
Xtr, Xva = make_X(D, "train", args.feats), make_X(D, "val", args.feats)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr = np.clip((Xtr - mu) / sd, -6, 6).astype(np.float32)
Xva = np.clip((Xva - mu) / sd, -6, 6).astype(np.float32)
all_seqs = seqs(Xtr, D["train_y"], D["train_sid"], np.ones(len(Xtr), bool))
out = []
for s in range(5):
    pv = predict(train(all_seqs, args, seed=1000 + s, epochs=cfg["e_star"])[0], Xva, D["val_sid"])
    r = score_inproc(events_from_probs(D["val_sid"], D["val_t"], pv, g), "val")
    out.append(round(r["ranking_score"]["f1_mean"], 4))
    print("seed", 1000 + s, out[-1], flush=True)
print(json.dumps({"single_seed_val_rank_f1": out, "mean": round(float(np.mean(out)), 4), "sd": round(float(np.std(out, ddof=1)), 4)}))
json.dump({"single_seed_val_rank_f1": out, "mean": float(np.mean(out)), "sd": float(np.std(out, ddof=1))}, open(OUT / "seed_var.json", "w"))

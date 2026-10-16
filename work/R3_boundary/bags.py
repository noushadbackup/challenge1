"""TCN-bag robustness: composite final_norep bundles for TCN bags 1/2 (seed offsets 10000/20000, the TCN members
of srcv/runs/tcn_bag{1,2}) + the norep HGB, re-tuned with uwbfall.pipeline.retune (clip stage, operating point,
OOF replay, NMS on TRAINING OOF only).     python3 bags.py bag1|bag2 [targets]
"""
from common import *  # noqa
import json
import shutil
from uwbfall.pipeline import retune

bag = sys.argv[1]
tg = [("val" if a == "val" else int(a)) for a in sys.argv[2:]] or TARGETS
for t in tg:
    nm = "official_train" if t == "val" else f"fold_{t}"
    src_n = BASE_RUNS / nm
    src_t = Path(f"/home/user/data_raw/srcv/runs/tcn_{bag}") / nm
    comp = BIG / "bags" / f"{bag}_src" / nm
    comp.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "meta.json", "pca.pkl", "hgb.pkl"):
        shutil.copy(src_n / f, comp / f)
    for f in ("tcn_0.pt", "tcn_1.pt", "tcn_2.pt", "tcn_norm.npz"):
        shutil.copy(src_t / f, comp / f)
    zn, zt = np.load(src_n / "oof.npz"), np.load(src_t / "oof.npz")
    assert (zn["sid"] == zt["sid"]).all() and (zn["fold"] == zt["fold"]).all()
    cfg = json.loads((src_n / "config.json").read_text())
    ct = json.loads((src_t / "config.json").read_text())
    cfg["tcn"]["seed_offset"] = ct["tcn"]["seed_offset"]
    (comp / "config.json").write_text(json.dumps(cfg, indent=1))
    np.savez_compressed(comp / "oof.npz", sid=zn["sid"], y=zn["y"], fold=zn["fold"], oof_hgb=zn["oof_hgb"],
                        oof_tcn=zt["oof_tcn"], oof_blend=zn["oof_blend"])
    out = BIG / "bags" / bag / nm
    retune(comp, train_sources(t), train_events(t), {}, out)
    print(f"[{bag}] {nm} retuned", flush=True)

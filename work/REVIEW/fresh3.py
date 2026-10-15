"""FG1 = FINAL fold bundles (srcv/runs/final_b0, bag 0) with the replay agreement gate set back to 0.1
(`predict --boundary '{"gate":0.1}'`). Targets: 'fresh' (seeds 9100+k), 'orig' (the 20 SR-CV streams), 'val'."""
import sys, time
from pathlib import Path
sys.path.insert(0, "/home/user/challenge1"); sys.path.insert(0, "/home/user/data_raw/abc2027_uwb_train_v1/src")
import srcv.core as C
target = sys.argv[1]
OUT = Path("/home/user/data_raw/review_fresh")
if target == "fresh":
    C.DATA = OUT
    _o = C._seed; C._seed = lambda f, p: 9100 + f if p == "test" else _o(f, p)
from srcv import bridge
from uwbfall.pipeline import predict, load_bundle
from uwbfall.sources import parse_sources
def bundle(path):
    B = load_bundle(path); B["cfg"] = {**B["cfg"], "boundary": {**B["cfg"]["boundary"], "gate": 0.1}}; return B
d = OUT / "preds_FG1" / target; d.mkdir(parents=True, exist_ok=True)
if target == "val":
    p = predict(bundle("/home/user/data_raw/srcv/runs/final_b0/official_train"), parse_sources("val")); p.to_csv(d / "val.csv", index=False)
else:
    for k in range(4):
        t0 = time.time(); p = predict(bundle(f"/home/user/data_raw/srcv/runs/final_b0/fold_{k}"), bridge.pipeline_sources(k, "test"))
        p.to_csv(d / f"fold_{k}.csv", index=False); print(target, k, len(p), round(time.time() - t0), flush=True)

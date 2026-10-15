"""Resume an SR-CV fold whose fit completed but whose predict was interrupted (container restart).

    python3 work/FINAL/resume_fold.py final_b0 1

Loads /home/user/data_raw/srcv/runs/<name>/fold_k (written by `srcv.run_pipeline`, train.log ends in 'fit done'),
predicts the fold's regenerated test streams exactly as run_pipeline does, writes srcv/results/<name>/fold_k.csv.
"""
import sys
import time

from srcv import bridge
from srcv import core as C
from srcv import scoring as S
import uwbfall

name, k = sys.argv[1], int(sys.argv[2])
bdir = C.DATA / "runs" / name / f"fold_{k}"
assert "fit done" in (bdir / "train.log").read_text(), "fit not complete"
t0 = time.time()
B = uwbfall.load_bundle(bdir)
for part in C.PARTS:
    bridge.seed_uwbfall_cache(k, part, B["cfg"])
pred = uwbfall.predict(B, bridge.pipeline_sources(k, "test"))
out = C.HERE / "results" / name
pred.to_csv(out / f"fold_{k}.csv", index=False)
rep = S.summary(S.score(out / f"fold_{k}.csv", k))
print(f"[{name}] fold {k} (resumed predict): rank F1 {rep['rank_f1']:.4f} tp/fp/fn@0.3 {rep['tp@0.3']}/"
      f"{rep['fp@0.3']}/{rep['fn@0.3']} ({time.time() - t0:.0f}s)", flush=True)

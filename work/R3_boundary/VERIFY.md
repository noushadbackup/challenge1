# VERIFY: R3 "boundary" (learned boundary head / wdet blend)

Independent verifier. I did not run the experiment, did not edit its files, and retrained nothing. My scratch scripts are in
`/tmp/claude-0/-home-user-challenge1/f892ab69-5d71-5ee6-a35c-b07836a0af8f/scratchpad/` (`rescore.py`, `indep.py`, `surf.py`).

**Verdict: CONFIRMED.** `wdet_cv` passes the promotion rule on bag 0. My independent re-implementation reproduces the
reported numbers, the per-target parameter choices and the CSVs exactly.

## 1. Re-scoring (srcv.score_pooled / srcv.promote / srcv.score_val)

| variant | pooled | delta [CI95] | P(delta>0) | per-fold delta | folds improved | val | rule |
|---|---|---|---|---|---|---|---|
| base norep | 0.9139 | | | | | 0.9373 | |
| wdet_cv | 0.9222 | +0.0083 [+0.0024, +0.0157] | 0.9952 | +0.0160 / +0.0114 / +0.0054 / 0.0 | 3 | 0.9424 | pass |
| wdet_sym | 0.9222 | +0.0083 [+0.0024, +0.0157] | 0.9952 | same as wdet_cv | 3 | 0.9424 | pass |
| wdet_cv bag 1 (vs its own base 0.9266) | 0.9284 | +0.0018 [-0.0060, +0.0098] | 0.662 | +0.0106 / -0.0033 / +0.0054 / -0.0051 | 2 | 0.9502 -> 0.9552 | fail |
| wdet_cv bag 2 (vs its own base 0.9147) | 0.9218 | +0.0071 [+0.0005, +0.0137] | 0.980 | +0.0106 / +0.0070 / +0.0107 / 0.0 | 3 | 0.9453 -> 0.9502 | pass |
| 3-bag seed-averaged | 0.9184 -> 0.9241 | +0.0057 [+0.0007, +0.0118] | 0.989 | +0.0124 / +0.0050 / +0.0071 / -0.0017 | 3 | 0.9443 -> 0.9493 | pass |

Every number matches the report. Other details also match: the wdet_cv pooled TP/FP/FN @0.3 is 240/21/13 (base 239/23/14), the median start
error is 0.195 s (base 0.04 s) and the mean tIoU is 0.782 (base 0.824).

## 2. Independent re-implementation (training-only selection, no leakage)

`indep.py` re-implements the blend from scratch. Its inputs are uwbfall `refine` (median of agreeing replay crops), then
s = (1-ws)·replay + ws·det on agreeing candidates (and the same for the end), then the dmin floor, then `to_preds` + `_nms`. The script:

- recomputes the 6x6 (ws, we) grid on each target's **training-OOF** candidates (`D_train.csv` = the bundle's
  `oof_candidates.csv` replayed on its training streams), scored against the **training** events only;
- takes the argmax using the stated tie-break (smaller ws+we, then smaller ws);
- applies the choice to `D_test.csv` and compares the result with `results/wdet_cv*/`.

Results for bag 0, bag 1 and bag 2, for all 5 targets:
- The chosen (ws, we) equal the reported ones. Bag 0: f0 (0.4,0.1), f1 (0.5,0.4), f2 (0.4,0.2), f3 (0.3,0.0), val (0.5,0.3).
- The CSVs are identical to the experiment's.
- `refine` + NMS without the blend reproduces the base fold/val CSVs (bag 0) and `dump_bagN/*/base_test.csv` (bags 1/2) exactly.
- Each bundle's `meta.train_streams` equals the target's training sources and is disjoint from its test streams. All D_train candidates lie in training streams and all D_test candidates in test streams.
- `oof_candidates.csv` comes from `uwbfall.pipeline._finish`, which decodes the inner-fold OOF blend. So the training-OOF bounds come from members that never saw those streams. The test candidates come from the final bundle members, which were trained only on the target's training streams.

The code is consistent with this. `grid2.py` reads only `D_train` plus the training events. `apply.py` reads `D_test` only to write predictions and score them. `bags.py` retunes the composite bundles with `uwbfall.pipeline.retune` on training data only. `select_b.py`, `fbsel.py` and `stack.py` select on D_train or inner-CV only, and `bfit.py` trains on training-stream events.

## 3. Refutation attempts

- **Is the chosen grid cell lucky?** As a verifier-only diagnostic, not a selection step, I scored all 36 fixed (ws, we) cells on the pooled SR-CV test folds (bag 0).
  Every cell with ws+we > 0 scores at or above base, from 0.9152 to 0.9235, and the surface rises roughly monotonically towards larger ws.
  The result is a broad plateau, not a spike, so the training-OOF argmax is not exploiting noise.
- **Multiplicity:** about 9 eligible variants were scored on test. The promoted family is the simplest of them, wdet_sym/fixed give the same result, and the whole surface is positive, so selection bias cannot explain the gain.
- **Seed dependence:** this is real. The gain is +0.0018 with P 0.66 on bag 1, and fold 3 is 0 or negative in every bag.
  The 3-bag average still passes (P 0.989, 3/4 folds, val +0.005). The gain is small: 1-2 events per fold, within the 0.004-0.013 seed-noise band of pooled scores, but it is paired against the same bag's base, which removes most of that noise.
- **Tie-break cost:** this is real and the report states it. The median start error rises from 0.04 to 0.195 s and the mean tIoU falls from 0.824 to 0.782.
  If the primary ranking F1 ties on the hidden test, this change loses the official tie-break.

## 4. Compliance

- `wdet_cv` uses only one candidate's detector bounds and that candidate's own-stream replay crops. There is no cross-stream or within-stream matching or alignment, no repeat counting and no new features. Only package data is used.
- An inherited grey zone, not introduced here: the base replay crops use `d` (dedup = drop a frame identical to the previous one before running the annotator). `reports/COMPLIANCE.md` already lists this as a grey zone of `final_norep`.
- The non-promoted dense model `b3` (`--fine2`) uses frame-difference energy smoothed by a 3-frame running max. This is not an explicit repeat statistic, but it is close to one (it partly hides zero-difference repeated frames). It is irrelevant to the promoted variant, which uses no model.
- Deployment needs (ws, we) chosen on the final bundle's own training OOF (the dump.py replay of `oof_candidates.csv`). Otherwise it is a fixed post-processing step.

## Summary

The numbers reproduce exactly, I found no leakage, and the variant is rule-compliant. It passes the rule on bag 0, bag 2 and the 3-bag mean, and fails on bag 1. The gain is small but robust over the
weight grid. The cost is a worse start-error tie-break.

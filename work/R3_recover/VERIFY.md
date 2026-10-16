# VERIFY: R3 "recover" (FN recovery in likely-fall clips)

Independent check (no retraining; 2 threads; only scoring and reading the saved pickles).
**Verdict: confirmed negative.** No variant passes the promotion rule. The base final_norep stays.

## 1. Re-scoring (srcv.score_pooled / srcv.promote vs work/STRICT/results/norep / srcv.score_val)

Base: pooled 0.9139, val 0.9373.

| variant  | pooled | TP/FP/FN@0.3 | delta   | CI95              | P(d>0) | folds | val    | promote |
|----------|--------|--------------|---------|-------------------|--------|-------|--------|---------|
| sec      | 0.9142 | 240/24/13    | +0.0003 | [-0.0057, 0.0058] | 0.538  | 1     | 0.9383 | False   |
| amax     | 0.9142 | 240/24/13    | +0.0003 | [-0.0057, 0.0058] | 0.538  | 1     | 0.9383 | False   |
| amax_rep | 0.9142 | 240/24/13    | +0.0003 | [-0.0056, 0.0059] | 0.543  | 1     | 0.9383 | False   |
| amax_seg | 0.9142 | 240/24/13    | +0.0003 | [-0.0057, 0.0058] | 0.538  | 1     | 0.9383 | False   |
| segcrop  | 0.9142 | 240/24/13    | +0.0003 | [-0.0057, 0.0058] | 0.538  | 1     | 0.9373 | False   |
| sec_rep  | 0.9121 | 239/24/14    | -0.0018 | [-0.0067, 0.0000] | 0.000  | 0     | 0.9383 | False   |
| raw      | 0.9139 | 239/23/14    | 0.0000  | [0, 0]            | 0.000  | 0     | 0.9453 | False   |

Every reported number matches. Folds 0 and 1 are byte-identical to the base for all variants: no config improved
their training OOF, so the base is kept. Folds 2 and 3 add 2-3 low-score rows each. raw changes only val, through
the official_train bundle's own choice (2 rows), so it has no SR-CV support.

## 2. Train-only tuning / leakage
- run.py: `RL.tune` scores on `RL.load(t, "train")` only. Those are the bundle's own training streams: oof.npz blend,
  oof_final.csv, oof_candidates.csv and the gt from `C.events(k,'train')` or `C.real_events('train')`.
  `RL.select` uses only that table. The test part (`test.pkl`) has no events. The chosen config is applied blind.
- Train and test stream sets are disjoint for all 5 targets (13 against 5, overlap 0). oof_final stream ids are a
  subset of the training streams.
- gather.py only reads the frozen base bundles. It asserts that the test predictions are byte-identical to the base
  CSVs and that gate x raw reproduces oof.npz. No model is retrained, so fold-test predictions come from bundles
  that never saw those streams.
- diag.py and chute04.py read the test labels, but only to count misses. run.py does not import them.
  - Minor caveat: the rule ideas were designed after looking at the test-side miss breakdown. This does not change a
    negative result.
- Small inaccuracy in the write-up: "averaged over each grid every rule is <= 0 on training OOF" is not quite true.
  The fold_3 amax, amax_rep and amax_seg grids average +0.0009 to +0.0017. All other grids average <= 0. This does
  not affect the conclusion.

## 3. Rule compliance
- Each rule (sec, amax, raw, segcrop) works on one stream's own p, gate, junction segments and rows.
  - The rep and seg boundaries run the base replay or propose_fall on that stream's own CIR crop.
  - Nothing is matched across streams, no repeated content is searched for, and no frame-repeat features are used.
- Only package data and the package's own code are used. No external models or data.

## 4. Pass claim
None claimed. None found.

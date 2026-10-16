# VERIFY: R3 "verify" (ResNet-18 image verifier), independent check

**Verdict: confirmed, as a clean negative.** No variant passes the promotion rule. Every reported number reproduces exactly, and I found no fold-test or val leakage in the tuning path and no rule violation.

I retrained nothing. Everything below was run with 2 threads or fewer, from the saved CSVs and probability files. My scratch scripts are in /tmp-like scratch only, and no experiment file was edited.

## 1. Re-scoring

I re-scored with `srcv.promote(A, B, val_A, val_B)`, `srcv.score_pooled` and `srcv.score_val`. A = `work/STRICT/results/norep/fold_{0..3}.csv` and `val.csv`. `results/base/` is byte-identical to those files.

| variant | pooled rank F1 | Δ | 95% CI | P(Δ>0) | folds improved (per-fold Δ) | val | F1@0.5 | FP@0.3 | promote |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.9139 | – | – | – | – | 0.9373 | – | 23 | – |
| r18_cut | 0.9265 | +0.0126 | [0, 0.0359] | 0.873 | 1 (0, +0.047, 0, 0) | 0.9373 | 0.8858 | 16 | False |
| r18x3_cut | 0.9283 | +0.0144 | [0, 0.0418] | 0.873 | 1 (0, +0.055, 0, 0) | 0.9373 | 0.8876 | 15 | False |
| r18_cut2 | 0.9302 | +0.0163 | [0, 0.0478] | 0.873 | 1 (0, +0.062, 0, 0) | 0.9373 | 0.8893 | 14 | False |
| r18_cut_s1 | 0.9211 | +0.0072 | [0, 0.0188] | 0.873 | 1 (0, +0.026, 0, 0) | 0.9373 | 0.8806 | 19 | False |
| r18_cut_s2 | 0.9229 | +0.0090 | [0, 0.0242] | 0.873 | 1 (0, +0.033, 0, 0) | 0.9373 | 0.8824 | 18 | False |

- All of these match the report and `promote.json` to 4 decimals.
- TP@0.3 stays at 239 in every variant.
- The val CSVs keep all 93 base rows, so val is unchanged.
- Every variant fails the folds clause (1/4 against 3 required) and the P clause (0.873 against 0.90). P is identical across variants because the gain sits in one subject cluster (fold 1).

## 2. Tuning and leakage

I re-implemented the `apply.py` tuning independently: the τ grid 0..0.7 step 0.025, the σ grid for cut2, and first-maximiser selection on the official rank F1 of the bundle's training-OOF `final` rows, scored against the training events. For all 5 variants × 5 sets, the τ/σ I got equal `tune.json`, and my output CSVs equal `results/<v>/*.csv` row for row.

Data-flow checks:
- **Training candidate streams:** in every set, all streams in `cands/<S>_train.csv` are training streams of that fold (or the 13 official train streams). There are 0 overlaps with the fold's test streams or val, and 0 shared subjects.
- **Bundle OOF inputs:** the stream ids in `oof.npz` and `oof_final.csv` contain no test or val stream.
- **Test rows:** `cands/<S>_test.csv` equals the base prediction rows exactly. The verifier only drops rows; it never edits times or scores.
- **`cnn.py full`:** trains only on `cands/<S>_train.csv` and crops, then predicts the test rows. Test labels (`hit`) are read only for the "analysis only" AUC and the `test_dropped_hits` count in `tune.json`; no decision uses them.
- **`cnn.py inner`:** GroupKFold(4) by subject over the training candidates only. τ is tuned on those inner-OOF probabilities.
- **Hyper-parameters:** 6 epochs, ±4 s window, lr, and the τ/σ grids are fixed constants in the code. The fold-0 screens (r18c6, r18hard, r18_oldcrops) were compared on the training-OOF AUC. r18c6 and r18_oldcrops also produced fold-0 test probabilities. The τ chosen in fold 0 is 0 in every case, so fold-0 test results equal the base whatever was chosen. I found no evidence that test or val drove a design choice, though I cannot prove the a-priori fixing from the code alone.
- **Caveat:** cut2 and the 3-seed mean were added after r18_cut had been scored. That is a multiple-variant concern, but it does not matter because nothing is promoted.

## 3. Rule compliance

- `maps.py` and `crops2.py` call `C.load_cir` per stream and compute frame-difference energy and amplitude from that stream only.
- The background is that stream's own per-range-bin median. This is per-stream normalisation over the stream's full length: offline, but not cross-stream.
- Nothing compares or aligns content across streams or with train/val segments. Nothing searches a stream for repeated portions, and no frame-repeat statistics are used. The "frame-difference" feature is |z_t − z_{t−1}|² motion energy, not a repeat detector.
- **External data:** only the challenge package plus torchvision ImageNet resnet18 weights (BSD-3, open source). These are loaded into a plain-torch re-implementation.

## 4. Pass claim

The experiment does not claim a pass, so there was nothing to refute. The fold-1 gain is real: 7 brushing-teeth FPs are dropped and 0 TPs are lost. But it comes from a single held-out subject and does not generalise across folds.

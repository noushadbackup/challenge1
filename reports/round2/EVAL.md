# Round 2: evaluation of C0–C4 on SR-CV, the replay × TCN puzzle, and the reference system

Author: evaluation engineer. Every number here comes from code run in this session (`work/R2_EVAL/`). All F1
values use the official `abc2027.evaluate` code (`--ignore-flagged`):
- SR-CV goes through `srcv.scoring`, which V6 shows equals the CLI.
- Val goes through the official CLI (`python -m abc2027.evaluate ... --streams streams/val --ignore-flagged`).
- Promotion decisions use `srcv.promote`: paired subject-cluster bootstrap with 5000 resamples, at least 3 of 4 folds improved, and val no worse than −0.02.

## TL;DR

* **REFERENCE SYSTEM = C3G3**: 0.75·TCN(3-seed bag) + 0.25·HGB, then annotator replay with the crop-agreement gate raised from 0.1 to **0.3**. It is now the uwbfall default (`DEFAULT_CONFIG` = `uwbfall/configs/reference.json`, checked by assert).
  * SR-CV pooled ranking F1 is **0.894** (3-bag mean, sd 0.009), against 0.886 for C3 (replay B).
  * Val is 0.9606 (3-bag mean). The shipped bag (bag 0) scores **0.9646, 65/0/2**, the same as before.
  * vs C3: Δ +0.008, P(Δ>0) = **1.000**, **4/4** folds improve. vs C2: P = 0.954, 3/4 folds. vs C0: P = 0.998, 4/4 folds.
* **The puzzle is resolved: replay B helped the blend and hurt the bare TCN on val, and both come from the same mechanism.**
  * Replay output is a *multi-modal* function of where the crop starts. Crops start at detector start − L, so a 0.2 s difference in detector start (TCN vs blend) can flip which motion burst `propose_fall` picks.
  * With gate 0.1, crops that locked onto a *different* burst (tIoU with the detector of only 0.1–0.3) enter the median. A 3-vs-3 split of the 6 crops then gives a meaningless midpoint.
  * Across 3 bags, the val effect of replay B on the bare TCN is −0.005 / 0 / 0. That is chance, not a TCN-specific defect. On SR-CV, replay B helps the TCN (+0.006) about as much as it helps the blend (+0.006). Neither passes the rule (P = 0.73 and 0.74).
  * Raising the gate to 0.3 removes most bad crops. Replay then passes the rule on **both** arms: C1→C4G3 P = 0.985 and C2→C3G3 P = 0.954.
* **What 3 seeds changed.** Seed-averaging over 3 TCN bags (2 new bags trained here) reversed two single-bag "passes": C1→C4 dropped from P = 0.952 to 0.731, and the jittered-crop fix C3J from +0.005 to +0.0004. **The TCN bag's own seed sd on pooled SR-CV F1 is 0.004–0.013**, about the size of every boundary-stage effect. Single-bag comparisons are not trustworthy.

## 1. Setup

**SR-CV.** 4 subject-disjoint folds and 20 regenerated test streams (8.24 h, 253 evaluable falls), from the SR-CV harness (`srcv/`).
- Each fold trains on its own 13 regenerated training streams.
- Val = 13 official train streams → 5 val streams.

**One fit per bag, all candidates derived from it.** The candidates differ only *after* training, so one fitted bundle per (fold, bag) serves all of them (`work/R2_EVAL/derive.py`):
- HGB, PCA and the TCN bag are fitted once.
- Each candidate's operating point (thr, smooth, min_dur, gap) is re-tuned on *that candidate's* training OOF: HGB OOF for C0, TCN OOF for C1/C4, blend OOF for C2/C3. The tuning uses the official evaluator with uwbfall's grid and first-max rule.
- Replay runs on that candidate's own candidates.

**Consistency checks** (byte-identical with `cmp`):
- C3 bag 0 == `srcv/results/uwb_default/fold_{0..3}.csv`, and val == `/home/user/data_raw/uwbfall_runs/val_default.csv`.
- C0 == `srcv/results/uwb_hgb_noreplay/fold_{0..3}.csv`.
- Variant R recomputed from the saved replay tables == derive outputs, for all 3 bags.
- `uwbfall predict` with the new reference config == the derived C3G3, on val (CLI, bag 0) and on SR-CV fold 0 (API).

**Seeds.** The rule asks to "seed-average ≥3 seeds", so this session trained **2 additional 3-member TCN bags**:
- `tcn.seed_offset` = 10000 and 20000, on all 4 SR-CV fold-trains and the official train (`work/R2_EVAL/fit_bag.py`, TCN-only fits).
- HGB is deterministic and is reused from the uwb_default bundles.
- Bag 0 = the original `uwb_default` bag (E4 seeds).
- The full E4 schedule was kept (8 epochs × 20 steps, 3 members). It ran about 20–25 min per fold-train with two fits in parallel. No step reduction was needed.

`srcv.promote` with a list of 3 prediction sets averages the bootstrap counts over bags, and per-fold deltas are bag means.

Candidates:
- C0: prototype HGB alone (detector bounds).
- C1: TCN 3-seed bag alone.
- C2: blend 0.75 TCN + 0.25 HGB.
- C3 = C2 + replay B (the previous uwbfall default).
- C4 = C1 + replay B.
- C3G3 / C4G3: the same with agreement gate 0.3.
- K / J / JK: other boundary fixes (§4).

## 2. Results (SR-CV pooled over 20 test streams, 3-bag mean; val with the official CLI)

| system | pooled rank F1 (3-bag mean ± sd) | 95% CI | per fold 0/1/2/3 | F1 @.1/.3/.5/.7 | TP/FP/FN @0.3 | FA/h | med start err | w/o P3_3 | val rank F1 3-bag mean (bag0) | val TP/FP/FN (bag0) |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 HGB alone, detector bounds | 0.8052 ± 0.0000 | [0.721, 0.865] | 0.819/0.833/0.857/0.702 | 0.856/0.835/0.724/0.465 | 203.0/30.0/50.0 | 3.64 | 0.280 s | 0.8326 | 0.8922 (0.8922) | 62/7/5 |
| C1 TCN bag alone, detector bounds | 0.8706 ± 0.0040 | [0.822, 0.910] | 0.865/0.817/0.895/0.908 | 0.917/0.897/0.798/0.469 | 234.7/35.3/18.3 | 4.29 | 0.347 s | 0.8639 | 0.9154 (0.9303) | 65/2/2 |
| C2 0.75 TCN + 0.25 HGB, detector bounds | 0.8796 ± 0.0085 | [0.826, 0.924] | 0.859/0.812/0.912/0.938 | 0.920/0.901/0.817/0.488 | 237.7/36.7/15.3 | 4.45 | 0.320 s | 0.8739 | 0.9370 (0.9394) | 64/1/3 |
| C3 = C2 + replay B (gate 0.1) | 0.8861 ± 0.0130 | [0.833, 0.930] | 0.864/0.844/0.877/0.960 | 0.908/0.896/0.854/0.711 | 236.0/38.0/17.0 | 4.61 | 0.040 s | 0.8795 | 0.9538 (0.9646) | 65/0/2 |
| C4 = C1 + replay B (gate 0.1) | 0.8763 ± 0.0057 | [0.826, 0.917] | 0.855/0.844/0.876/0.931 | 0.898/0.890/0.841/0.693 | 232.7/37.0/20.3 | 4.49 | 0.040 s | 0.8690 | 0.9138 (0.9254) | 62/5/5 |
| **C3G3 = C2 + replay, gate 0.3 (REFERENCE)** | **0.8942 ± 0.0092** | [0.843, 0.935] | 0.869/0.847/0.891/0.970 | 0.915/0.904/0.863/0.713 | 238.0/35.7/15.0 | 4.33 | 0.040 s | 0.8881 | **0.9606 (0.9646)** | 65/0/2 |
| C4G3 = C1 + replay, gate 0.3 | 0.8849 ± 0.0074 | [0.837, 0.924] | 0.871/0.841/0.892/0.938 | 0.910/0.897/0.847/0.698 | 234.3/35.0/18.7 | 4.25 | 0.040 s | 0.8782 | 0.9284 (0.9453) | 64/3/3 |
| C3K clean crops only | 0.8885 ± 0.0126 | [0.837, 0.930] | 0.864/0.850/0.891/0.950 | 0.912/0.896/0.857/0.722 | 236.3/38.0/16.7 | 4.61 | 0.040 s | 0.8820 | 0.9572 (0.9646) | 65/0/2 |
| C3J 12 jittered crops | 0.8866 ± 0.0189 | [0.834, 0.929] | 0.857/0.847/0.875/0.967 | 0.907/0.897/0.855/0.711 | 236.3/37.7/16.7 | 4.57 | 0.040 s | 0.8800 | 0.9622 (0.9697) | 65/0/2 |
| C3JK (J + K) | 0.8861 ± 0.0143 | [0.835, 0.928] | 0.856/0.849/0.884/0.957 | 0.908/0.896/0.854/0.711 | 236.0/38.0/17.0 | 4.61 | 0.040 s | 0.8795 | 0.9589 (0.9646) | 65/0/2 |
| C4K | 0.8805 ± 0.0038 | [0.831, 0.921] | 0.859/0.846/0.890/0.929 | 0.902/0.890/0.849/0.712 | 232.7/37.0/20.3 | 4.49 | 0.040 s | 0.8735 | 0.9251 (0.9353) | 63/4/4 |
| C4J | 0.8784 ± 0.0041 | [0.831, 0.918] | 0.857/0.846/0.876/0.936 | 0.902/0.892/0.842/0.700 | 233.0/36.7/20.0 | 4.45 | 0.040 s | 0.8712 | 0.9237 (0.9502) | 64/3/3 |
| C4JK | 0.8814 ± 0.0035 | [0.834, 0.920] | 0.859/0.846/0.887/0.936 | 0.903/0.889/0.852/0.712 | 232.3/37.3/20.7 | 4.53 | 0.040 s | 0.8744 | 0.9300 (0.9453) | 64/3/3 |

How to read the table:
- **Means and CI.** Counts and F1 are 3-bag means. The CI is the subject-cluster bootstrap on the bag-averaged counts.
- **C0 sd.** C0 has sd 0 because the HGB is deterministic and identical in every bag.
- **Per-bag numbers.**
  - Pooled SR-CV, bags 0/1/2:
    - C1: .866/.872/.874
    - C2: .878/.872/.889
    - C3: .885/.874/.900
    - C3G3: .889/.889/.905
  - Val, bags 0/1/2:
    - C3: .965/.947/.950
    - C3G3: .965/.952/.965
- **Ranking components.** At tIoU 0.1/0.3, replay costs a little (C2 → C3: F1@0.1 .920 → .908). At 0.5 it adds a lot (.817 → .854), and it cuts the first tie-break, median |start error|, from 0.32 s to 0.04 s.
- **Gate 0.3.** It removes most of the 0.1/0.3 cost (F1@0.1 .915) and keeps the 0.5 gain (.863).

## 3. Promotion rule (3 bags; `work/R2_EVAL/out/promote_*.json`)

| A → B | Δ pooled [95% CI] | P(Δ>0) | folds improved (Δ per fold) | val A → B (3-bag mean) | w/o P3_3 A → B | rule |
|---|---|---|---|---|---|---|
| C2 → C3 (replay B on blend) | +0.0065 [−0.013, +0.025] | 0.743 | 3/4 (+.005 +.032 −.035 +.022) | .937 → .954 | .874 → .880 | **fail** (P) |
| C1 → C2 (add HGB, no replay) | +0.0089 [−0.003, +0.022] | 0.921 | 2/4 (−.007 −.005 +.018 +.031) | .915 → .937 | .864 → .874 | **fail** (folds) |
| C1 → C4 (replay B on TCN) | +0.0057 [−0.011, +0.023] | 0.731 | 2/4 (−.010 +.027 −.019 +.023) | .915 → .914 | .864 → .869 | **fail** |
| C0 → C3 | +0.0807 [+0.020, +0.169] | 0.997 | 4/4 | .892 → .954 | .833 → .880 | **pass** |
| C4 → C3 (add HGB, with replay) | +0.0097 [+0.001, +0.019] | 0.989 | 3/4 (+.009 −.000 +.001 +.029) | .914 → .954 | .869 → .880 | **pass** |
| **C3 → C3G3 (gate 0.1 → 0.3)** | **+0.0082 [+0.003, +0.016]** | **1.000** | **4/4** (+.005 +.003 +.014 +.010) | .954 → .961 | .880 → .888 | **pass** |
| C4 → C4G3 | +0.0086 [+0.003, +0.016] | 0.998 | 3/4 | .914 → .928 | .869 → .878 | pass |
| C2 → C3G3 (replay with gate 0.3 vs none) | +0.0146 [−0.002, +0.033] | 0.954 | 3/4 (+.010 +.035 −.021 +.032) | .937 → .961 | .874 → .888 | pass |
| C1 → C4G3 | +0.0143 [+0.002, +0.028] | 0.985 | 3/4 | .915 → .928 | .864 → .878 | pass |
| C0 → C3G3 | +0.0889 [+0.029, +0.177] | 0.998 | 4/4 | .892 → .961 | .833 → .888 | pass |
| C3 → C3J (12 jittered crops) | +0.0004 [−0.004, +0.007] | 0.512 | 2/4 | .954 → .962 | .880 → .880 | fail |
| C3 → C3K (clean crops) | +0.0024 [−0.003, +0.008] | 0.780 | 2/4 | .954 → .957 | .880 → .882 | fail |
| C3 → C3JK | +0.0000 | 0.496 | 2/4 | .954 → .959 | | fail |
| C3G3 → C3K | −0.0058 | 0.046 | 2/4 | .961 → .957 | | fail |
| C4 → C4K / C4 → C4JK / C4 → C4J | +.004 / +.005 / +.002 | .970 / .917 / .799 | 3 / 4 / 3 of 4 | | | pass / pass / fail (TCN-only arm; not used) |

**Single-bag vs 3-bag.** With bag 0 alone (`work/R2_EVAL/out/bag0/`), the results looked different:
- C1→C2 passed (P 0.958, 3/4) and C1→C4 passed (P 0.952, 3/4).
- C2→C3 failed (P 0.775, 2/4).
- C3→C3J looked like +0.005.

Seed-averaging moved all four. This is the main methodological finding: **the TCN bag's seed noise (sd 0.004–0.013 pooled) is as large as the boundary effects**, so the rule's ≥3-seed clause is essential.

**Why C3G3 is the reference.**
- It is the best pooled candidate, and it passes the rule against C3 (the incumbent default), C2, C0 and, via C3 → C3G3, C4.
- C3 itself never passed against C2. C3G3 does (P 0.954), so the reference now includes replay on rule-passing evidence, not only for its tie-break benefit.
- The HGB member is justified *given replay* (C4 → C3: P 0.989). Without replay, C1 → C2 fails the fold clause (2/4).
- The val clause holds for every promoted step.

**Caveat on how gate 0.3 was found.**
- An agreement gate was one of the levers suggested beforehand ("gate on … replay agreement"). The value 0.3 was not swept: it equals the primary matching threshold.
- I first saw G3 as a diagnostic on bag 0 (+0.004). The two new bags (+0.000 → +0.016 and +0.005) are independent *seed* replications, but they use the *same* SR-CV subjects.
- Five boundary variants were tried (K, KD, G3, J, JK), so there is mild selection. G3's P = 1.000 with 4/4 folds is far from the threshold, so I consider the promotion robust to that.

## 4. The replay × TCN puzzle: mechanism, with per-event evidence

**The puzzle (INTEGRATION, bag 0).** On val, replay B took the blend from 0.934 to 0.965 but the bare TCN from 0.930 to 0.925. Re-derived here: C1 0.9303 → C4 0.9254, and C2 0.9394 → C3 0.9646.

**Evidence 1: the val effect is not systematic.** Across 3 bags, C1 → C4 on val is −0.005, 0.000 and 0.000. On SR-CV, replay B helps the TCN (+0.0057) about as much as the blend (+0.0065), and neither passes. Interval statistics do not differ either:

| bag 0, GT-matched candidates | median det start − GT start | median det width | crop proposals carrying an annotator review reason |
|---|---|---|---|
| SR-CV, C3 (blend) | −0.10 s | 2.0 s | 39.4 % |
| SR-CV, C4 (TCN) | −0.10 s | 1.8 s | 39.9 % |

No width gate is indicated: binning the bag-0 flips by detector width or agreement gives no clean cut (`work/R2_EVAL/mech.py`).

**Evidence 2: the 4 val events that replay B lost for the TCN** (bag 0; `work/R2_EVAL/out/evidence_official.csv`; GT is the official non-flagged event):

| stream | TCN det | GT | what the 6 crops (L_R) returned | B median → tIoU | gate-0.3 result |
|---|---|---|---|---|---|
| val_02 | 88.20–91.80 | 89.52–92.36 | `loc/det` crops start at 86.2 s and pick an *earlier* burst, 86.9–88.7 (`start_not_visible`, 4 crops). The `detq` crops are clipped at a junction and get it right, 89.6–92.2 (2 crops). | 88.21–90.44 → 0.22 | 89.59–92.29 → 0.95 (the blend started at 89.2 s, so its crops started later and all 6 found the right burst) |
| val_00 | 173.20–175.80 | 173.00–174.58 | Bimodal: the `_3` crops give 174.88–176.48 and the `_5` crops give 172.62–175.56, 3 vs 3. | Median of the two modes, 173.75–176.02 → 0.27 | A 3-vs-3 split survives gate 0.3, so this one is not fixed (12 jittered crops fix it: 0.54) |
| val_00 | 51.40–53.20 | 50.52–52.52 | 3 crops 50.48–50.74 (`event_under_1s`, cut by the gate); 3 crops 52.02–53.20 (the post-fall burst) | 52.02–53.20 → 0.19 | Detector bounds kept → 0.42 |
| val_01 | 1315.40–1316.80 | 1313.00–1316.98 | All 6 crops give 1315.96–1316.9 (`event_under_1s`) | 0.25 | 0.25 (unfixable by gating) |

**The mechanism.**
- `propose_fall` places the fall at *the last motion burst before the new static profile inside the crop*. The crop is [det start − L, det end + R]. So the result is a step function of the detector bounds: shift the crop start or end by a fraction of a second and it jumps to another burst.
- Typical alternatives are the pre-fall activity, if the crop starts too early, and post-fall movement or the next stream segment, if the detector interval is wide and extends past the fall.
- Gate 0.1 lets a crop that found the neighbouring burst into the median, because it still overlaps a wide detector interval by ≥10 %. The median of 6 then lands on the wrong burst (majority) or between bursts (3-vs-3).
- TCN and blend intervals differ by ±0.2–1 s on individual events, so which events get hit is effectively random. That is why the val sign differed between the two arms.

**SR-CV evidence for the fix** (`work/R2_EVAL/out/g3_flips.txt` and `g3_events.csv`, 3 bags):
- Of 736 GT-matched C3 candidates, 40 had at least one crop in the 0.1–0.3 agreement band.
- Moving the gate from 0.1 to 0.3 changes the tIoU class of only those:

| threshold | gains | losses |
|---|---|---|
| @0.1 | 6 | 1 |
| @0.3 | 10 | 4 |
| @0.5 | 14 | 7 |

- For C4 (TCN), the gains/losses are 9/0, 11/7 and 14/10.
- The remaining losses are events where *all* crops sit in the low-agreement band because the detector interval is much wider than the fall. There the replay was right, and the gate falls back to the detector bounds.
  - Example: cv2_test_00 at 512.6–515.8 s against a GT of 514.94–515.94 s.

**Fixes tested and rejected.**
- **K (clean crops, rejected).** Use only crops whose proposal has no review reason, since every scored GT proposal had none. Result: +0.0024, P = 0.78.
- **J (12 jittered crops, rejected).** Add `{loc,det,detq}{1.5,2.5}_4d` to break 3-vs-3 ties. It fixes val_00@173 and val_02 and gives val 0.9697 on bag 0. Result: +0.0004, P = 0.51.
- **Lesson.** Both looked good on bag 0, val or both. Neither holds across seeds.

## 5. Changes made to the packages (documented, minimal)

| file | change | effect on existing behaviour |
|---|---|---|
| `uwbfall/config.py` | New key `tcn.seed_offset` (default 0), added to every CV and final TCN seed. `cv_seed_base` alone cannot re-seed a bag, since CV member 0 of fold k always gets seed k (an issue the SR-CV engineer flagged). | None at 0. Used for bags 1 and 2 (10000, 20000). |
| `uwbfall/pipeline.py` | Uses `seed_offset` in the 2 `train_tcn` seed expressions. | None at 0. |
| `uwbfall/config.py` | **`boundary.gate` default 0.1 → 0.3** (the promoted change). | New fits and predicts use gate 0.3. Existing bundles keep their stored gate unless overridden at predict time. |
| `uwbfall/configs/reference.json` | New file: the REFERENCE SYSTEM written out explicitly. Asserted equal to `load_config(None)` and to `default.json` (`{}`). | |
| `uwbfall/configs/integration_repro.json` | Now also pins `boundary.gate: 0.1`, so it still reproduces INTEGRATION exactly. | |
| `uwbfall/configs/replayB_gate01.json` | New file: the previous default (C3). | |
| `uwbfall/cli.py` | `train --config` defaults to `configs/reference.json`. `predict --boundary <json or file>` overrides the bundle's boundary section at predict time; replay does not depend on training. | Without `--boundary`, predict behaves exactly as before. |
| `uwbfall/Makefile` | `CONFIG ?= configs/reference.json` | |

srcv was not modified. Its `promote` already accepts a list of seeds.

**Not changed.**
- The 1-ulp float parsing in `uwbfall.parse_sources` (SR-CV open issue). It only duplicates cache keys, and the exact path `srcv.pipeline_sources` was used throughout.
- `--override` for model sections. Retraining is the correct path there.

**The shipped models do not need retraining for the gate.** The bundles at `/home/user/data_raw/uwbfall_models/{default_train13,...}` store gate 0.1, so predict with `--boundary uwbfall/configs/reference.json`, or retrain. Verified on bag 0:
- `python3 -m uwbfall predict --models /home/user/data_raw/srcv/runs/uwb_default/official_train --split val --boundary uwbfall/configs/reference.json` scores 0.9646, 65/0/2, median start 0.05 s, F1@.7 .803.
- Its output is byte-identical to the derived C3G3.

## 6. Limitations

- **SR-CV size.** Folds are subject-disjoint but there are only 24 fall and 8 ADL subjects.
  - Fold 3's D3 result rests on Test_P3_3. All promoted steps also hold without it (the "w/o P3_3" column).
  - Folds are regenerated with the public generator. The unshipped test rebuild may differ.
- **Selection.** Gate 0.3 was promoted after looking at 5 boundary variants (see §3). One more independent check would be a fresh SR-CV fold seed (new stream compositions), which this session had no compute left for.
- **Frozen hyper-parameters.** The blend weight 0.75, E* = 8 epochs and the operating-point grid are frozen from E4. They were not re-selected per fold, apart from the operating point, which is tuned on each fold's training OOF.
- **D2 recall.** C3G3 still has about 36 FPs and 15 FNs @0.3 on SR-CV; per the SR-CV report, about half the FPs come from unseen ADL pieces (ADL_12, teeth brushing). This is the next lever (W3/W5), not boundaries.
- **Compute.** About 2.4 h wall: 2 extra bags × 5 fits, run in parallel at about 20–25 min each, plus derive/replay/score.
  - Peak RSS per process was 1.3 GB, and threads were 2 per process.
  - With two fits in parallel the machine was fully used (4 threads).

## 7. Reproduce

```bash
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
python3 work/R2_EVAL/fit_bag.py tcn_bag1 10000        # and tcn_bag2 20000 (TCN-only bags; HGB reused from uwb_default)
python3 work/R2_EVAL/derive.py <bag> [0,1,2,3,official]  # C0..C4 preds + replay tables
python3 work/R2_EVAL/jitter.py <bag>                    # extra crops for J variants
python3 work/R2_EVAL/variants.py uwb_default,tcn_bag1,tcn_bag2   # K/KD/G3/J/JK (+ byte check of R)
python3 work/R2_EVAL/score_all.py table <systems> uwb_default,tcn_bag1,tcn_bag2
python3 work/R2_EVAL/score_all.py promote C3 C3G3 uwb_default,tcn_bag1,tcn_bag2
python3 work/R2_EVAL/mech.py; python3 work/R2_EVAL/evidence.py official   # per-event mechanism tables
```
Model bundles are under `/home/user/data_raw/srcv/runs/{uwb_default,tcn_bag1,tcn_bag2}/{fold_k,official_train}`.
Replay tables and probabilities are in `work/R2_EVAL/cache/`, which git ignores. Prediction CSVs are in `work/R2_EVAL/preds/<bag>/<system>/` (2.1 MB). JSON summaries are in `work/R2_EVAL/out/`.

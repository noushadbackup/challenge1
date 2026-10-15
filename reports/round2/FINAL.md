# Round 2 FINAL: frozen config, clean SR-CV run, final models, submission dry run

Author: round-2 integrator. Every number below comes from code run in this work package (`work/FINAL/`) and was scored with the official
`abc2027.evaluate` code (`--ignore-flagged`):
- SR-CV is pooled over 20 regenerated test streams (253 evaluable falls), via `srcv.scoring`.
- Val is scored with the official CLI.
- Promotion uses `srcv.promote` through the R2_EVAL tooling: a paired subject-cluster bootstrap with 5000 resamples, plus the fold clause and the val clause.

**Resume note.** A container restart at about 05:16 killed the first attempt's background lanes. Its partial outputs were re-checked before reuse:
- The final train+val bundle was already complete (`train_final rc=0`). Its files match `models/final/MANIFEST.sha256`.
- SR-CV folds 0 and 2 were complete. Their CSVs are byte-identical to I1's G7N4 bag 0.
- Fold 1's fit was complete but its predict was cut off. Prediction was resumed from the completed bundle (`work/FINAL/resume_fold.py`).
- Fold 3 and official-train → val were re-run from scratch.
- The physics fallback bundles had never been started, so they were trained in this run.

## TL;DR

* **Frozen config:** `uwbfall/configs/final.json` = the reference = `DEFAULT_CONFIG` (`tests/check_configs` OK). It holds every promoted change:
  1. round 1: 0.75 TCN (3 seeds) + 0.25 prototype HGB;
  2. R2_EVAL: annotator replay with crop-agreement gate 0.3;
  3. I1: segment-classifier gate p·(0.7 + 0.3·P(fall clip)) and a 4 s time NMS.

  I2 and I3 promoted nothing, so none of their flags are on. In particular, I3's closest miss XHTPFN (P = 0.886) and I2's masked TCN (1 bag only) are **not** deployed.
* **Clean run:** `srcv.run_pipeline final_b0 --config final.json` fits from features end to end. It reproduces I1's G7N4 bag-0 predictions **byte-for-byte** on all 4 SR-CV folds and on val, so the retune path that I1 used for its decisions equals a fresh fit.
* **Final table:** SR-CV pooled ranking F1 is **0.9221 ± 0.003** (3 TCN bags; CI [0.883, 0.953]); the clean bag 0 alone is 0.9183. The round-1 integration system C3 scores 0.8861. **C3 → FINAL: Δ +0.036 [+0.020, +0.054], P = 1.000, 4/4 folds.** False positives @0.3 drop from 38.0 to 20.0 and false alarms/h from 4.61 to 2.43. Official val is 0.9524 (bag 0; 64/2/3 @0.3); the 3-bag mean is 0.9517, against C3's 0.9538 (−0.002, within tolerance).
* **Final models:** trained on all 18 labelled streams in `/home/user/data_raw/uwbfall_models/final`. The operating point (0.4, 3, 0.8, 0.6) comes from the bundle's own 4-fold training OOF, whose rank F1 is 0.9389 after replay. A runnable 3.7 MB copy with sha256 is in `models/final/`.
* **Dry run:** the val streams were copied as test_00..04, and the literal command `python -m uwbfall predict --models … --split test` was run end to end. Official `validate_predictions` passes, `check_submission` OK, the drift monitors raise 0 alerts, and timing and RSS are given in §4. The physics-only fallback is trained, scored on val (§5) and rehearsed too.

## 1. Frozen config

`uwbfall/configs/final.json` (13 lines) spells out every non-default-sensitive section explicitly. `check_configs` asserts that `final.json`,
`reference.json` and `DEFAULT_CONFIG` resolve to the same config (hash `204d32de158f1814`, as recorded in every bundle trained with it).

| stage | setting | promoted by | evidence |
|---|---|---|---|
| members | HGB on the 64 prototype features (300 iterations), plus 3 TCN seeds on 112 channels (8 epochs) | round 1 (E4/INTEGRATION) | EVAL.md: C0 → C3 P 0.997; C4 → C3 P 0.989 |
| blend | 0.75 TCN + 0.25 HGB | round 1 | EVAL.md |
| clip gate | p·(0.7 + 0.3·P(fall clip)); LightGBM segment classifier on 18 non-fingerprint features, subject-grouped OOF | I1 | REF → G7N4 P 1.000, 4/4 folds |
| decode | operating point from each bundle's own training OOF (official evaluator) | (method) | |
| boundaries | `abc2027.annotate.propose_fall` replay on 6 crops, median of crops with tIoU-agreement ≥ **0.3**, else detector bounds | R2_EVAL | C3 → C3G3 P 1.000, 4/4 folds |
| final NMS | at most one row within 4 s of a higher-scored row | I1 | REF → N4 P 1.000, 4/4 folds |
| transductive ADL-duplicate mask | **off** | rule compliance | |

Not deployed:
- I2's zone features, ignore-clip weights, cluster aggregation, containment fallback and selector: all failed the rule.
- I3's hit model and threshold: the best variant reached P = 0.886, below the 0.90 bar.
- The masked TCN: screened on one bag only, so not eligible.

## 2. Clean run and final table

`make srcv` runs `srcv.run_pipeline final_b0 --config uwbfall/configs/final.json` (4 folds), then `--val-only` (13 official train streams → val), then `work/FINAL/score.py link|check|table|promote`.
- FINAL bag `uwb_default` is this clean run (`srcv/results/final_b0`, bundles in `/home/user/data_raw/srcv/runs/final_b0`).
- FINAL bags `tcn_bag1` and `tcn_bag2` are I1's G7N4 predictions from the seed-offset-10000/20000 TCN bags. They come from the same config through the retune path, which I1's `parity_clip_retune` and the byte-identity below show to be equivalent.
- C3 is the round-1 integration system: 0.75 TCN + 0.25 HGB + replay B with gate 0.1, from `work/R2_EVAL/preds/<bag>/C3`.

`score.py check`: fold_0..3.csv and val.csv of the clean run are **byte-identical** to `work/I1_clip/preds/uwb_default/G7N4`.

SR-CV pooled over 20 test streams (253 evaluable falls), 3-bag means; val from the official CLI:

| system | pooled rank F1 (± sd over bags) | 95% CI | folds 0/1/2/3 | F1 @.1/.3/.5/.7 | TP/FP/FN @0.3 | FA/h @0.3 | med. start err | val, 3-bag mean | val, bag 0 (TP/FP/FN @0.3) |
|---|---|---|---|---|---|---|---|---|---|
| C3 (round-1 integration) | 0.8861 ± 0.013 | [0.833, 0.930] | .864/.844/.877/.960 | .908/.896/.854/.711 | 236.0/38.0/17.0 | 4.61 | 0.04 s | 0.9538 | 0.9646 (65/0/2) |
| **FINAL (final.json)** | **0.9221 ± 0.003** | [0.883, 0.953] | .918/.889/.906/.975 | .946/.933/.887/.730 | 238.7/20.0/14.3 | 2.43 | 0.04 s | 0.9517 | **0.9524 (64/2/3)** |

Per bag, pooled SR-CV (folds) and val:

| bag | C3 | FINAL |
|---|---|---|
| uwb_default (E4 seeds; FINAL = clean run) | 0.8851 (.885/.825/.873/.960), val 0.9646 | 0.9183 (.921/.873/.909/.972), val 0.9524 |
| tcn_bag1 (+10000) | 0.8736, val 0.9474 | 0.9235, val 0.9502 |
| tcn_bag2 (+20000) | 0.8996, val 0.9495 | 0.9244, val 0.9524 |

Promotion rule, C3 → FINAL (`work/FINAL/out/promote_C3_vs_FINAL_*.json`):

| bags | Δ pooled [95% CI] | P(Δ>0) | folds improved (Δ per fold) | val A → B | without Test_P3_3 | decision |
|---|---|---|---|---|---|---|
| 3 (seed-averaged) | **+0.0361 [+0.020, +0.054]** | **1.000** | **4/4** (+.054, +.045, +.029, +.015) | 0.9538 → 0.9517 | 0.880 → 0.918 | **PASS** |
| bag 0 only (clean run) | +0.0332 [+0.015, +0.054] | 1.000 | 4/4 (+.036, +.048, +.036, +.012) | 0.9646 → 0.9524 | 0.880 → 0.915 | PASS |

**Reading.**
- Almost all of the gain is false-positive removal: FP @0.3 drops 38 → 20 at equal recall (TP 236 → 238.7).
- Val falls by 0.012 on bag 0: 65/0/2 → 64/2/3 @0.3, i.e. two extra FPs and one extra miss. I1.md §4 traces these events. Over 3 bags it falls by 0.002. Both are within the rule's 0.02 tolerance, and I2.md reports that its null control alone moved val by 0.007.
- Val has 67 evaluable falls; SR-CV has 253 falls from 20 subjects held out by subject. The decision rests on SR-CV.

## 3. Final models (train + val, 18 streams)

`make train-final` = `python -m uwbfall train --config uwbfall/configs/final.json --train-splits train,val --out /home/user/data_raw/uwbfall_models/final`

- 18 streams, 478 events, 148 145 blocks.
- Stream-grouped 4-fold OOF for HGB and TCN; the final members are refit on all 18 streams.
- Clip stage: junction threshold 1.011 (fall-junction recall 0.949 on the training streams); segment classifier on 2079 segments, subject-grouped OOF AUC 0.980.
- The **operating point is picked on the bundle's own training OOF** with the in-process official evaluator: (thr 0.4, smooth 3, merge 0.8, min-dur 0.6). Training-OOF rank F1 is 0.9315 for the detector and 0.9389 after replay and NMS, with 239/10/14 @0.3. Val streams enter only as training data and OOF folds; no choice was made on val scores.
- Wall time was 869 s (TCN 668 s) at 2 threads, with peak RSS 1.6 GB. Environment: python 3.13.16, numpy 2.5.3, sklearn 1.9.1, torch 2.14.1+cpu.
- `drift_ref.json`, the label-free monitor reference from the bundle's training streams, was added with `python -m uwbfall driftref`.

Shipped copy: `models/final/` holds config, meta, pca, hgb, clip, tcn_0..2, tcn_norm and drift_ref, 3.73 MB in total.
- `MANIFEST.sha256` is in `sha256sum -c` format and verifies both the copy and the bundle in data_raw (`make verify-models`).
- `manifest.json` records the source, config hash, training streams, operating point, OOF scores and environment.
- Not shipped, and regenerable by `make train-final`: oof.npz, oof_*.csv, train.log.

**Reproducibility check (this run).** `make train-final FINAL=/home/user/data_raw/uwbfall_models/final_refit_check` was re-run from scratch with the current code (git `b3716b8` plus the working tree): 1064 s, 1.64 GB.
- `sha256sum -c models/final/MANIFEST.sha256` on the refit gives **9/10 files bit-identical**: config, pca, hgb, clip, tcn_norm, tcn_0..2 and drift_ref.
- `meta.json` differs only in `git_rev`, `timings` and `peak_rss_gb`. The operating point and OOF score are identical.
- The check bundle was deleted afterwards. Log: `work/FINAL/logs/train_final_refit_check.log`.

## 4. Submission dry run (val streams as test_00..04)

Fixture: `make fake-test` (new target) copies the 5 val streams to `/home/user/data_raw/fake_test/streams/test/stream_0X.h5`. The package is untouched; ids become test_00..04 from the folder name.
- It also symlinks the package `src` and writes `streams_events_fake.csv`, the val labels with renamed ids, which are used only for the in-sample wiring check below.
- The final bundle was **trained on these streams**, so any score on them is in-sample and only proves the plumbing.

| step | command | result |
|---|---|---|
| literal release command, cold caches | `ABC_ROOT=/home/user/data_raw/fake_test UWBFALL_CACHE=<empty> python -m uwbfall predict --models /home/user/data_raw/uwbfall_models/final --split test --out …/submission_split_test.csv` | rc 0. 119 rows, 5 streams. Official `validate_predictions` OK inside predict. **260 s wall, 1.60 GB peak RSS**, about 50 s/stream of feature extraction (114 MB of feature cache). Log: `logs/dryrun_cold_split_test_r2.log` |
| Makefile path, warm cache | `make dry-run` = `fake-test` + `verify-models` (sha256, 10/10 OK) + `predict --sources dir:…:test` + `check-test` | rc 0. **SUBMISSION CHECK: OK** (header, ids test_00..04, official validator, 18–34 rows per stream, 36.6–69.4 events/h). 58 s wall in total, 1.12 GB. Log: `logs/dryrun_make_r2.log` |
| equivalence | `cmp` of the two CSVs | byte-identical, and identical to the first attempt's dry-run CSV |
| drift monitors (printed by predict) | 16 monitors | **0 ALERT**: events/h 58.1 (ref 55.0), junctions/h 243 (ref 250), repeat share 0.235 (0.236), near-junction share 0.78 (0.84), raw level inside the training range on all 3 radars, PSI ≤ 0.009 everywhere |
| drift sanity (`make checks`) | `tests/check_drift` on val features | clean → no alert. CIR ×10 → level + PSI alerts. Repeats removed → repeat-share + rep_frac alerts. Junction-blind → junction-rate alert. `check_configs` OK |
| in-sample wiring check | official evaluator, `streams_events_fake.csv` | 0.9701 (66/1/1 @0.3). In-sample, **not** a performance estimate |
| fallback path | `make predict-test-fallback TEST_DIR=…fake_test/streams/test` (now also runs `verify-fallback`) | rc 0, 116 rows, SUBMISSION CHECK OK, 0 drift alerts, 35 s |
| `make reproduce-val` | final.json on 13 train streams → val (bundle = clean run `official_train`, linked as `uwbfall_models/final_train13`) | 0.9524 (64/2/3), val CSV byte-identical to I1 G7N4 bag 0, 55 s |

The drift monitors are label-free and computed within the predicted streams (`uwbfall/drift.py`), against `drift_ref.json` from the bundle's training streams:
- predicted events/h;
- detected junctions/h;
- the share of predictions starting ≤ 5 s after a junction;
- the sample-and-hold repeat share;
- per-radar PSI of block log motion energy, log frame-difference energy and repeated-frame fraction (10 training-quantile bins);
- the per-radar median raw-energy level.

Two or more ALERT lines make predict print `RECOMMEND physics-only fallback`. The thresholds are heuristic (§6).

## 5. Physics-only fallback

`uwbfall/configs/fallback_physics.json` uses:
- only the prototype "physics" columns (static/diff/jump, 25 features). It drops the EMA-junction, t_since/t_until and segment features and the PCA fingerprint;
- no clip gate;
- replay with the 2 junction-free `loc` crops;
- the same HGB + TCN blend, gate 0.3 and 4 s NMS.

It is meant for a test writer whose clip/junction structure differs from the package generator's. It was trained in this run:

| bundle | training streams | training-OOF rank F1 (detector → final) | TP/FP/FN @0.3 (OOF) | official val |
|---|---|---|---|---|
| FINAL (`uwbfall_models/final`) | train+val (18) | 0.9315 → 0.9389 | 239/10/14 | (in-sample) |
| fallback (`uwbfall_models/fallback_physics_train13`) | train (13) | 0.9223 → 0.9369 | 177/7/9 | **0.9239** (59/1/8; F1 @.1/.3/.5/.7 .945/.929/.898/.756) |
| fallback (`uwbfall_models/fallback_physics`) **= release-day switch** | train+val (18) | 0.9109 → 0.9200 | 234/13/19 | (in-sample) |
| FINAL config on train (13), for comparison | train (13) | n/a | n/a | 0.9524 (64/2/3) |

- The fallback costs about 0.03 on val, mostly recall (59 vs 64 TP). It is not run on SR-CV.
- Its train+val bundle has a sha256 copy in `models/fallback_physics/` (2.65 MB), checked by `make verify-fallback`.
- Fits took 880–950 s each at 2 threads, with peak RSS 1.66–1.82 GB.
- Decision rule (SUBMISSION.md §4): submit FINAL unless ≥ 2 ALERTs include a structural monitor (repeat share or junction rate).

## 6. Limitations

* **Selection on the shared SR-CV streams.** Every round-2 decision used the same 20 regenerated SR-CV test streams: gate 0.3, I1's gate floor and NMS gap, and the 30+ variants of I2/I3. C3 → FINAL is far above the bar (P = 1.000, CI lower bound +0.020), so selection cannot explain it. The *absolute* SR-CV level 0.922 is still mildly optimistic. A fresh SR-CV regeneration with new composition seeds was not run (time box).
* **Val went down slightly** (bag 0: 0.9646 → 0.9524; 3-bag: 0.9538 → 0.9517). This is accepted by the rule and within val noise. If the lead weights val more, I1.md documents the conservative alternative: `configs/C3G3.json` plus the 4 s NMS. It was not re-run here.
* **Seed noise.** The deployed bundle is a single 3-member TCN bag (E4 seeds). Bag-to-bag SD of pooled SR-CV is 0.003 for FINAL. A larger TCN ensemble (e.g. 3 bags × 3 seeds) was not evaluated as a system and is not deployed.
* **Bit-exactness depends on the environment.** TCN outputs depend on the torch build and thread count, so reproduce with `OMP_NUM_THREADS=2` and torch 2.14.1+cpu.
* **The fallback is weaker and is a switch only** (§5). The drift alert thresholds are heuristics, checked only on synthetic drift (`tests/check_drift.py`) and on clean val.
* Remaining errors (I1/I2/I3) are about 20 FPs and 14 FNs @0.3 on SR-CV. The FNs are mostly Chute04 falls that get no candidate at all, plus mislocalised rows in unmatched fall clips.

## 7. Files

* `uwbfall/configs/final.json`, `uwbfall/configs/fallback_physics.json`, `uwbfall/drift.py`, `uwbfall/tests/{check_submission,check_drift,check_configs}.py`
* `work/FINAL/`:
  * `score.py`: table and promotion;
  * `resume_fold.py`;
  * `ship.py`: copy to models/ plus manifest;
  * `rss.py`: wall time and peak-RSS wrapper;
  * `lane*.sh`: run scripts;
  * `logs/`;
  * `out/`: `table.json`, `promote_C3_vs_FINAL_{3bags,bag0}.json`, dry-run and fallback outputs;
  * `preds/`: symlinks.
* `srcv/results/final_b0/`: clean-run fold CSVs, val.csv, `summary.json` (pooled, rebuilt from the 4 fold CSVs) and `val_summary.json`.
* `models/final/`: the shipped bundle copy. `SUBMISSION.md` is the release-day runbook. `Makefile` targets: features, train-final, predict-test, reproduce-val, srcv, plus checks, dry-run and fallback.

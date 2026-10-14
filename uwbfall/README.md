# uwbfall: submission pipeline (PLAN v2, work package W6)

`uwbfall` is an installable, deterministic Python package. It runs the **current best system** end to end,
from raw UWB streams to a validated submission CSV:

> prototype 64 features + E4 range bands (112 channels) → prototype HGB + 3-seed TCN bag → blend 0.75·TCN + 0.25·HGB
> → prototype post-processing (operating point chosen on training OOF) → E2 annotator replay, config B, for boundaries.

All code is ported from `analysis/` and `work/E1,E2,E4`. Nothing is imported from `work/` at run time; only the
parity tests import from there. The package only reads the challenge package and never modifies it.

**Verification headline.** Trained on the 13 train streams only and scored on val with the official CLI
(`--ignore-flagged`), the INTEGRATION configuration reproduces **0.9646 (TP/FP/FN 65/0/2 @0.3)** exactly.
- The val CSV is byte-identical to `work/INTEGRATION/out/val_tcn75_proto25_replayB.csv`.
- The train OOF probabilities of the HGB and of the TCN bag are bit-equal to E4's `probs_hgb.npz` and `probs_tcn_all.npz`.
- Details are in the Verification section below.

## Quick start

```bash
cd /home/user/challenge1
export PYTHONPATH=/home/user/challenge1          # or: pip install ./uwbfall
# train (13 train streams; ~19 min on 2 threads with a cold TCN, features cached after the first run)
python3 -m uwbfall train --config uwbfall/configs/default.json --train-splits train --out /home/user/data_raw/uwbfall_models/default_train13
# predict (runs the official validate_predictions before writing the CSV)
python3 -m uwbfall predict --models /home/user/data_raw/uwbfall_models/default_train13 --split val  --out /home/user/data_raw/uwbfall_runs/val.csv
python3 -m uwbfall predict --models /home/user/data_raw/uwbfall_models/default_train13 --split test --out /home/user/data_raw/uwbfall_runs/test.csv
# official scoring (python -m abc2027.evaluate ... --ignore-flagged)
python3 -m uwbfall score --pred /home/user/data_raw/uwbfall_runs/val.csv --split val
# exact reproduction of the INTEGRATION result (HGB early_stopping="auto"):
make -C uwbfall verify-repro
```

Other commands:
- `python3 -m uwbfall extract --sources train,val,test` warms the feature cache.
- `--train-splits train,val` trains on train and val together, for the final test submission.
- `python3 -m uwbfall predict ... --candidates cand.csv` also writes the detector candidates from before the replay.

### Python API

```python
import pandas as pd
from uwbfall import fit, predict, load_config, split_sources, sources_from_composition, VirtualSource
ev = pd.read_csv(".../streams_events.csv", dtype=str)
bundle_dir = fit(split_sources("train"), ev, load_config({"hgb": {"early_stopping": False}}), "/path/model_dir")
df = predict(bundle_dir, split_sources("val"))            # DataFrame stream_id,t_start,t_end,score,t_alarm
# virtual streams (regenerated with the official planner, never written as h5):
srcs = [VirtualSource(f"p{seed}_{i:02d}", st.segments) for i, st in enumerate(plan.streams)]
df = predict(bundle_dir, srcs)
```

`fit(train_sources, events_df, config, out_dir)` takes `events_df=None` when every source is virtual. The labels
then come from the writer's own events.

## Architecture

| module | role | ported from |
|---|---|---|
| `sources.py` | `H5Source` (a `stream_XX.h5`) and `VirtualSource` (segments rebuilt in memory). Both expose `radar(r)`, `crop(i0,i1)`, `n_frames`, `content_key()` and `release()`. Also `build_stream` (the generator writer replica) and factories `split_sources`, `dir_sources`, `sources_from_composition` and `parse_sources`. | `e1lib.build_stream` (= `build_streams._write_stream`) |
| `extract.py` | Per-radar extractor (`e_raw, e_mot, e_diff, mot_map, prof_02`) and the feature cache. | `e1lib._one_radar` (= `analysis/extract_features.py`) |
| `features.py` | Prototype block features, structure features, train-only PCA fingerprint, the 64 "full" columns in prototype order, the 48 band channels, block labels. | `analysis/prototype_detector.py`, `E4/data.py` |
| `models.py` | HGB and the dilated TCN (same modules, sampler, RNG call order and loss as E4). | `E4/tcn.py`, `E4/hgb_ref.py` |
| `decode.py` | `to_events`, the 60-point operating-point grid, tuning on OOF with the official evaluator code. | prototype / `E4/lib.py` |
| `boundary.py` | Annotator replay on distinct frames (`abc2027.annotate.propose_fall`), the crops `loc/det/detq L_R d`, refine (gate, median, floor) and clipping. | `E2/s2_replay.py`, `E2/s3_select.py` |
| `scoring.py` | Thin wrappers over `abc2027.evaluate`: `load_ground_truth`, `validate_predictions`, `evaluate`, and the CLI subprocess. | package code |
| `pipeline.py` | `fit`, `load_bundle`, `predict`. | INTEGRATION |
| `cli.py` | `train / predict / score / extract`. | |

**Fit** runs these steps:
1. Features are extracted one stream at a time and cached.
2. The PCA fingerprint is fitted on the train streams only (every 5th block, 16 components).
3. Folds are `GroupKFold(4)` by stream.
4. HGB OOF and the final HGB are fitted.
5. TCN: standardise on all train blocks (label-free). For each fold k and bag member s, train with seed `100·s+k` for 8 epochs. OOF is the seed mean. Then 3 final members are trained with seeds `1000+s`.
6. The blend `0.75·TCN + 0.25·HGB` is formed on OOF. The operating point (thr, smooth, min_dur, gap) is the first maximum of the official ranking F1 over the grid on train OOF.
7. With `report.oof_replay`, the boundary replay also runs on the OOF candidates and is saved as `oof_final.csv`. This is the pooled OOF a later SR-CV needs.

**Predict** handles one stream at a time:
1. Features (cache), then HGB, then the TCN bag (mean of 3 members), then the blend.
2. `to_events`, rounded to 3 decimals as in the reference pipeline, gives the candidates.
3. The replay reads the raw crops (h5 slice, or the in-memory virtual stream) and refines the bounds.
4. Rows are clipped to the stream duration and, in the CLI, validated with `validate_predictions` before the CSV is written.

**Bundle** (`<model_dir>/`) contains:
- `config.json`, `meta.json` (operating point, OOF scores, decode grid table, folds, train content keys, timings, versions, git rev).
- `pca.pkl`, `hgb.pkl`, `tcn_{0,1,2}.pt` (0.5 MB each), `tcn_norm.npz`.
- `oof.npz` (per-block OOF of each model and the blend), `oof_candidates.csv`, `oof_final.csv`, `train.log`.
- About 5 MB in total. Model directories live outside git under `/home/user/data_raw/uwbfall_models/`.

**Feature cache.** Files are `/home/user/data_raw/uwbfall_cache/feat_<sha256(content_key | EXTRACTOR_VERSION)[:24]>.npz`, about 28 MB per 30-min stream.
- For h5 sources, `content_key` is the sha256 of the `cir` dataset bytes. It is memoised in `h5_hash_index.json` by path, size and mtime.
- For virtual sources, it is the sha256 of the writer version plus the full-precision segment list and segment file sizes.
- The cache is safe to delete. Other agents may share it, since keys are content-based.

**Resources.** Every stage holds at most one stream's raw CIR.
- Peak RSS was 1.07 GB for train and 0.84 GB for predict from h5. Predicting from virtual streams with a cold cache peaked at 1.85 GB.
- Threads are fixed at `runtime.threads = 2`. `uwbfall/__init__.py` sets `OMP/MKL/OPENBLAS_NUM_THREADS` before numpy and torch load, and `threadpoolctl` caps native pools inside fit/predict.
- Without this cap, the HGB ran a 4-thread OpenMP team and slowed down more than 5x on the shared machine. That was observed in this session, and the fix brought it back to normal speed.

## Config schema (JSON, deep-merged over `config.DEFAULT_CONFIG`; unknown keys are rejected)

| section.key | default | meaning |
|---|---|---|
| `runtime.threads` | 2 | torch, OpenMP and BLAS threads. TCN bits depend on it, and E4 used 2. |
| `runtime.deterministic` | true | `torch.use_deterministic_algorithms(True)` |
| `runtime.cache_dir`, `runtime.use_cache` | null, true | Feature cache location (default `/home/user/data_raw/uwbfall_cache`, or `$UWBFALL_CACHE`). |
| `features.proto_set` | "full" | physics, structure or full (prototype column sets) |
| `features.pca_components`, `pca_stride` | 16, 5 | Train-only static-amplitude fingerprint |
| `features.bands`, `use_bands` | E4 8 bands, true | Range-band channels (48). Set false for 64 channels. |
| `labels.flagged` | "positive" | Flagged falls are positives (prototype/E4). Reserved for loss masking (W4). |
| `cv.n_folds` | 4 | GroupKFold by stream on the training sources, used for the operating-point OOF |
| `hgb.*` | prototype params; **`early_stopping: false`** | `max_iter 300, lr 0.06, max_leaf_nodes 31, l2 1.0, random_state 0`. `"auto"` reproduces INTEGRATION (`configs/integration_repro.json`). |
| `tcn.*` | E4 | `hidden 64, stages 1, dilations 1..64, drop 0.2, in_drop 0.1, noise 0.05, lr 1e-3, wd 1e-2, win 1000, batch 16, steps 20, epochs 8 (E*), seeds 3, cv_seed_base 100, final_seed_base 1000, clip_z 6` |
| `blend.w_tcn` | 0.75 | Weight of the TCN. One model can be disabled with `hgb.enabled` / `tcn.enabled`. |
| `decode.grid` | 5×3×2×2 | thr, smooth, min_dur, gap |
| `decode.fixed` | null | `[thr, smooth, min_dur, gap]` skips the OOF search |
| `decode.one_event_per_clip` | false | **reserved** (W3). Raises NotImplementedError if enabled. |
| `boundary.method` | "replay" | replay or none |
| `boundary.crops` | B's 6 crops | `loc2_3d, loc2_5d, det2_3d, det2_5d, detq2_3d, detq2_5d` (grammar `(loc\|det\|detq)L_R[d]`) |
| `boundary.gate`, `min_ok`, `dmin` | 0.1, 1, 1.0 | Agreement gate tIoU(replay, detector), minimum agreeing crops, duration floor (end-extension only) |
| `boundary.ds/de/pds/pde` | 0 | Start/end offsets, for replayed and detector-bound rows |
| `boundary.dedup` | true | (documentation; the `d` crop suffix controls it per crop) |
| `boundary.junction_q`, `junction_q_recall` | 97, 95 | Junction percentile thresholds for `det` / `detq` crops |
| `score.hit_model` | null | **reserved** (W5). The score is the peak smoothed blend probability. |
| `transductive.adl_duplicate_mask` | false | **reserved, must stay OFF** (rules ledger) |
| `report.oof_replay` | true | Also replay the OOF candidates (`oof_final.csv`, OOF score in `meta.json`) |

To add an option, add its key to `DEFAULT_CONFIG` with the "off" value that preserves current behaviour and branch
on it in `pipeline.py`. If it is a placeholder, list it in `NOT_IMPLEMENTED`.

## Verification (all numbers from runs in this session; val scored with the official evaluator CLI)

### 1. Replica / parity checks (`uwbfall/tests/`)

| check | result |
|---|---|
| `parity_features`: extractor keys of all 18 published streams vs `/home/user/data_raw/feats/*.npz` | **bit-equal** (all 5 keys used) |
| `parity_features`: X_proto (64), X_band (48), labels, column order and GroupKFold folds vs `E4_deep_model/cache/blocks.npz` | **bit-equal** for train (111,268 blocks) and val |
| `parity_regenerated`: virtual streams from a regenerated plan (E1 seed 1001, streams 0 and 4) through the uwbfall extractor vs E1 bank features | **bit-equal** |
| `parity_virtual`: published val streams rebuilt from `streams_composition.csv` (3-decimal rows) vs h5 | val_00/01/02/04 **bit-equal**; val_03: 338 of 88,526 frames differ (one ADL piece at 1136-1144 s is off by one frame) |
| `parity_models`: train OOF of HGB ("auto") and of the TCN bag vs E4 `probs_hgb.npz` / `probs_tcn_all.npz` | **bit-equal** (max abs diff 0) |
| operating point on blended OOF | (0.5, 5, 0.8, 0.2), OOF 0.9428: same as E4 `compare_tcn_all.json` |
| val candidates vs `E4/out/val_ens_tcn_all_w0.75.csv` | identical (92 rows) |
| val final CSV vs `INTEGRATION/out/val_tcn75_proto25_replayB.csv` | **byte-identical** (`cmp`) |
| re-running predict | byte-identical CSV |

The val_03 difference is the half-frame ambiguity documented by E1. The CSV prints `source_t0/t1` with 3 decimals.
`sources_from_composition` resolves most cases from the published stream times alone: the block length, and for falls
the event offset. A residual case remains when both ends of an ADL row sit on half frames. Predictions from the
virtual val streams still score **0.9646**; only one score digit differs (val_03 0.839 vs 0.84). Regenerated streams
use full-precision segments and have no ambiguity.

### 2. Train on the 13 train streams, predict val, official CLI

| config | train OOF rank F1 (detector → +replay) | **val rank F1** | F1 @0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | FA/h | med \|start err\| |
|---|---|---|---|---|---|---|
| `integration_repro.json` (HGB early_stopping "auto") | 0.9428 → 0.9410 | **0.9646** | .985/.985/.924/.818 | 65/0/2 | 0.00 | 0.06 s |
| `default.json` (HGB early_stopping **false**, PLAN §2.3) | 0.9410 → 0.9410 | **0.9646** | .985/.985/.924/.803 | 65/0/2 | 0.00 | 0.05 s |

Notes:
- The replay lowers train OOF ranking F1 slightly (0.9428 → 0.9410), but F1@0.7 rises from 0.676 to 0.763 and the median start error drops from 0.20 s to 0.03 s. On val it adds +0.030.
- This is the unresolved interaction PLAN §0 lists; it must be settled by SR-CV, not here.

**Why the default differs from the reproduction.** PLAN §2.3 requires HGB `early_stopping=False`.
- With `"auto"`, sklearn holds out a stratified 10 % validation split (random_state 0), so every HGB fits on 90 % of the blocks.
- Both settings are deterministic, but they are different models.
- The difference is a change of the HGB member, so under the promotion rule it is *not* validated by this val number. In this session the two settings give the same val ranking F1 (0.9646, 65/0/2). The default has F1@0.7 0.803 vs 0.818 and a median start error of 0.05 s vs 0.06 s. The HGB OOF probabilities differ by up to 0.60 per block (Pearson 0.986), and train OOF is 0.9410 vs 0.9428. That is all within val noise (about ±0.04).
- TCN weights are byte-identical between the two training runs, a cross-run determinism check.
- The default is `false` because the task spec and PLAN §2.3 ask for it. `integration_repro.json` keeps the exact reference.

### 3. Wall time (4-CPU machine shared with another agent; 2 threads)

| step | wall |
|---|---|
| feature extraction, cold cache (hash + extract + compress) | 35-56 s per 30-min stream (18 streams ≈ 13 min) |
| `train` on 13 streams, warm cache | **18.8 min** (repro) / **16.7 min** (default): features 20 s, HGB OOF + final 100 s, TCN 4-fold × 3-seed CV + final bag 930 s, operating-point search 1 s, OOF replay 70 s |
| `predict` val, 5 streams, warm cache | **31-33 s** (h5) |
| `predict` 5 streams, cold cache (virtual val, includes build + extraction) | **108 s**, peak RSS 1.85 GB |

A 5-stream test split predicts in about 2-4 min with a cold cache.

## Honest limitations

- **Nothing here is a model improvement.** The package reproduces the candidate system, and the 0.9646 is still a val-only number. Promotion needs the pooled SR-CV OOF (W1). `fit` already saves per-block OOF and replayed OOF events to make that easy.
- The operating point and the OOF are computed with stream-grouped folds on the training sources, as in E4. Train OOF is leaky (ADL reuse, shared subjects), so the threshold may be slightly optimistic. For SR-CV, pass subject-disjoint sources to `fit`. A custom fold assignment is not exposed yet; `cv.n_folds` is GroupKFold by stream.
- The E* (8 epochs) and the blend weight (0.75) are frozen from E4, not re-selected by `fit`. Training on train+val uses the same values.
- **TCN bits depend on the thread count and the torch build.** Bit-exactness was shown with torch 2.14.1+cpu, 2 threads and sklearn 1.9.1. On another machine, expect tiny numeric differences in the TCN, not a different system.
- The h5 content hash reads each file once (about 400 MB); later runs use the memo.
- The replay needs raw CIR at prediction time: an h5 slice or the in-memory virtual stream. With virtual sources, each predicted stream is rebuilt once, which takes a few seconds and about 0.4 GB.
- `boundary.crops` supports `loc/det/detq` crops only. E2's frame-refined `detr*/detqr*` variants were not ported, because config B does not use them.
- Unchanged generator-artefact risks (PLAN §6): junction/structure features and the distinct-frame replay assume the test writer behaves like `_write_stream`.

## Files

- `uwbfall/`: package (`__init__, __main__, cli, config, paths, sources, extract, features, models, decode, boundary, scoring, pipeline`), `pyproject.toml`, `Makefile`, `configs/{default,integration_repro}.json`.
- `uwbfall/tests/`: `parity_features.py`, `parity_virtual.py`, `parity_regenerated.py`, `parity_models.py`, `smoke_virtual_fit.py` (fit + predict on virtual sources only).
- Outside git: `/home/user/data_raw/uwbfall_models/{repro_train13,default_train13}`, `/home/user/data_raw/uwbfall_runs/*.csv|json`, `/home/user/data_raw/uwbfall_cache/`.

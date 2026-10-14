# SR-CV: subject-disjoint regenerated cross-validation (PLAN v2 W1)

SR-CV is the validation harness behind the promotion rule (PLAN v2 §2). It pools the train and val **subjects** and
splits them into 4 folds. For each fold it **regenerates** streams with the official package generator:

- held-out **test** streams use split `test` settings;
- **train** streams for the remaining subjects use split `train` settings.

No raw stream is ever written to disk. A stream is rebuilt in memory from its composition rows by a bit-exact
replica of the package writer, and only extracted features are cached.

Every number in this README comes from code in this directory, run in this session:

- logs: `results/*.log`
- JSON: `results/verify.json`, `results/verify_bridge.json`, `results/baseline_prototype/summary_full.json`

All scores use the official `abc2027.evaluate` code with `--ignore-flagged` semantics.

## 1. What was built

| file | role |
|---|---|
| `core.py` | Pool of planning rows (train+val). Subjects. Fold plans via `prepare_split` + `finish_plan` (density 57.1864). Fold CSVs. `Source` objects (virtual or h5). `load_cir` and `load_feats` with a guarded feature cache. |
| `replica.py` | `write_stream`: bit-exact replica of `scripts/build_streams._write_stream`. `plan_stream`: the same index arithmetic on segment lengths only, so CSVs are made without reading radar data. `extract`: the E1/analysis feature extractor. The writer and extractor are vendored from E1 and re-verified here. |
| `build.py` | Deterministic fold search, then `folds.json`, then `/home/user/data_raw/srcv/fold_k/{train,test}_{composition,events}.csv`. |
| `cache_feats.py` | Pre-computes the feature cache, using 2 workers at about 1.5 GB RSS each. |
| `scoring.py` | `score`, `score_pooled`, `score_val` (official evaluator), `cluster_counts` (subject attribution), `bootstrap_ci`, `promote` (the promotion rule). |
| `compare.py` | CLI wrapper of `promote` for two prediction directories. |
| `bridge.py` | Builds `uwbfall.VirtualSource` lists from SR-CV folds and seeds uwbfall's feature cache from srcv's cache. |
| `run_pipeline.py` | Runs `uwbfall.fit`/`predict` on all 4 folds, then pooled score + CI. |
| `baseline_prototype.py` | The end-to-end baseline (prototype recipe, 'full' features) on all 4 folds. |
| `verify.py` | Verification suite V1–V7 (section 3). |

Data locations:

- `srcv/folds.json`: in git; also copied to `/home/user/data_raw/srcv/folds.json`.
- `/home/user/data_raw/srcv/fold_k/`: the CSVs, about 0.5 MB per fold.
- `/home/user/data_raw/srcv/feats/<stream_id>.npz`: feature cache, plus a `.json` sidecar holding the plan hash. About 2.3 GB for all 72 streams.
- `/home/user/data_raw/srcv/runs/`: uwbfall model bundles.

All of these are outside git.

## 2. Folds

There are 24 fall subjects: 8 per dataset, 16 from train and 8 from val. There are 8 ADL subjects, identified by the
trailing number of `activities/<Act>_<n>.json`: 0, 4, 5, 6, 9, 10, 11 and 12.

Each subject is held out exactly once. Every fold holds out 2 D1 + 2 D2 + 2 D3 fall subjects, with their own
Chute/Autre/Empty files, plus 2 ADL subjects. This is the shape of the official val/test split.

The fold search (`build.search_folds`, rng 2027, 200k random assignments) minimises:

- the variance of evaluable test falls per fold;
- plus half the D1 and D2 variances separately;
- with at most 2 original-val fall subjects per fold;
- and the two val ADL subjects in different folds.

Test_P3_3 is the only D3 subject with evaluable falls (16 of 20). It is held out in fold 3, which therefore gets
fewer evaluable D1/D2 falls.

| fold | D1 | D2 | D3 | ADL | test: streams / h / falls / evaluable | train: streams / h / falls | seeds (train/test) |
|---|---|---|---|---|---|---|---|
| 0 | P1_5, P1_6 | P2_1, P2_9 | P3_5, P3_6 | 4, 6 | 5 / 2.067 / 120 / 62 | 13 / 6.166 / 358 | 7100 / 7200 |
| 1 | P1_7, P1_9 | P2_3, P2_5 | P3_2, P3_7 | 11, 12 | 5 / 2.051 / 119 / 62 | 13 / 6.185 / 359 | 7101 / 7201 |
| 2 | P1_3, P1_8 | P2_6, P2_7 | P3_1, P3_9 | 9, 10 | 5 / 2.052 / 119 / 63 | 13 / 6.186 / 359 | 7102 / 7202 |
| 3 | P1_1, P1_2 | P2_2, P2_8 | **P3_3**, P3_8 | 0, 5 | 5 / 2.067 / 120 / 66 | 13 / 6.164 / 358 | 7103 / 7203 |

Every fold-part comes out at 58.0–58.1 falls/h, the same as the published 58.08. Pooled, there are **253 evaluable
test falls** across 20 test streams (8.24 h).

For comparison, the real val split has 5 streams, 2.05 h and 67 evaluable falls. The fold training sets have the
same size as the official train split: 13 streams, about 6.2 h, 18 fall subjects and 6 ADL subjects, at `adl_uses=3`.

### How a fold is generated

```
pool rows = scripts/build_streams._groups() logic over train+val  (fall rows from streams_events.csv:
            t_start/t_end = source_t_start/end, quality, needs_review, fall_type, dataset; t_last from the
            segments/*.npy frame count; ADL/other duration_s = frames*0.02)
prepare_split("test"|"train", falls, adl, others, rng=np.random.default_rng(seed))  # official code
finish_plan(bundle, 57.1864)                                                          # official code
plan_stream(stream_id, segments)  -> composition + events CSV (+ exact float columns)
```

- `part="test"` uses split `"test"` settings: `adl_uses=2`, and flagged falls (quality low or needs_review) get `ignore=true`.
- `part="train"` uses split `"train"` settings: `adl_uses=3`, and `ignore=false` everywhere, as in the published train events.

## 3. Verification (`python3 -m srcv.verify`, `results/verify.json`; all checks passed)

| check | result |
|---|---|
| **V1** pool rows vs the official `build_streams._groups()` (train + val) | 478 fall rows, 120 ADL rows and 24 other rows. All fields are identical (t_start, t_end, quality, needs_review, fall_type, dataset, path; ADL/other duration_s). |
| **V2** public-seed replay: `prepare_split(split, our rows)` with the public seed reproduces the published fall clip windows | `source_t0`: **478/478 identical**, which proves t_start, the margins and the rng consumption. For `source_t1` (the clip end), `t_last = n·0.02`, as in `_groups`, gives 9 mismatches (train 5, val 4), every one a clip clipped at `t_last`. **`t_last = (n−1)·0.02` gives 477/478** (val 1 remains, off by 1 frame). The published plan therefore used a `t_last` one frame shorter than the aligned segment, and SR-CV uses `(n−1)·0.02` (`core.T_LAST_MODE`). The effect is at most 1–2 frames at the end of about 2% of the clips. |
| **V3** `plan_stream` (lengths only) on the published composition | All 18 published streams match: every composition time, every events.csv field (478 events) and every stream length T equals the published h5. |
| **V4** `write_stream` vs published h5 | `val_04` and `train_00` are **bit-identical on all 3 radars**. The vendored extractor is bit-identical to the team cache (`/home/user/data_raw/feats`) on all 7 keys for both streams. |
| **V5** regenerated `cv0_test_04` | `write_stream` (data) gives the same composition, events and T as the stored fold CSVs (`plan_stream`). |
| **V6** `score_val` vs the official CLI | The full JSON report is identical for `sample_submission.csv` (0.315091) and the prototype val CSVs: full 0.883951, physics 0.790323, structure 0.875912. |
| **V7** `bridge.pipeline_sources` (uwbfall) | n_frames are equal for all 8 fold-parts. On `cv0_test_04`, the cir is bit-equal, uwbfall's extractor equals srcv's on its 5 keys, and the events are equal. |
| scorer self-check | `cluster_counts` asserts that its per-subject TP/FP/FN sum to the official `evaluate()` counts at every θ, on every call. |

The package was never written. All runs used `PYTHONDONTWRITEBYTECODE=1`, and `build_streams` is imported only for
the read-only `_groups()`.

## 4. API

```python
import sys; sys.path.insert(0, "/home/user/challenge1")
import srcv

srcv.folds()                    # list of 4 dicts: fold, test_subjects{D1,D2,D3,ADL}, seeds, generated stats
srcv.sources(k, "train"|"test") # list[srcv.Source]  (virtual streams cv{k}_train_XX / cv{k}_test_XX)
srcv.real_sources("train"|"val"|"test")  # list[srcv.Source] for the published h5 streams
srcv.events(k, part)            # DataFrame, official streams_events.csv columns + subject
srcv.composition(k, part)       # DataFrame, official composition columns + exact times/flags/subject
srcv.durations(k, part)         # {stream_id: seconds} (= T/50, what evaluate.stream_durations reads)
srcv.load_cir(src)              # (3, T, 368) float32, bit-exact (virtual: ~0.4 GB per 30-min stream, ~5-60 s)
srcv.load_feats(src)            # dict of the 7 cached feature arrays (computed + cached if missing)

srcv.score(pred, k)             # official report dict on fold k's test streams (pred: DataFrame or CSV path)
srcv.score(pred, k, part="train")   # same on the fold's training streams (for OOF post-processing choice)
srcv.score_pooled({k: pred_k})  # official report on all 20 test streams (pooled OOF); a single DataFrame
                                #   holding cv{k}_test_* rows for all folds also works
srcv.score_val(pred)            # official report on the real val streams (== CLI json, V6)
srcv.scoring.bootstrap_ci(preds)          # pooled rank F1 + subject-cluster bootstrap 95% CI
srcv.promote(A, B, val_A=..., val_B=...)  # promotion rule (below)
```

### `Source` format (what a pipeline consumes)

`srcv.Source` is a frozen dataclass. `src.as_dict()` gives the same content as a plain dict, and every srcv function
accepts either form.

```python
Source(
  stream_id  = "cv0_test_03",      # 'cv{k}_{part}_{i:02d}' | real: 'val_02'
  kind       = "virtual",          # 'virtual' (rebuilt from segments) | 'h5'
  split      = "test",             # generator settings used: 'train' | 'test' (real: package split)
  fold       = 0,                  # None for real streams
  n_frames   = 88626,              # T ; cir shape (3, T, 368)
  duration_s = 1772.52,            # T / 50
  segments   = ({"kind": "adl", "path": "activities/Manger_4.json", "label": "Manger", "dataset": "",
                 "source_t0": 47.43857590629841, "source_t1": 64.02381128790732,   # full float precision
                 "event_t0": None, "event_t1": None, "fall_type": "", "quality": "",
                 "needs_review": False, "ignore": False}, ...),     # writer input, stream order
  h5_path    = "",                 # real streams only
  feats_path = "/home/user/data_raw/srcv/feats/cv0_test_03.npz",
)
```

`segments` has the same attributes as `abc2027.streams.Segment`. That means `abc2027.streams.Segment(**seg)` or
`uwbfall.sources.Seg(**seg)` can be built from any row.

### Using it with the uwbfall pipeline

`uwbfall` already defines `VirtualSource` and `sources_from_composition`. `srcv.bridge` feeds those directly, using
full-precision times, so no half-frame nudging is needed. This path is verified by V7.

```python
from srcv import bridge
import srcv, uwbfall
preds = {}
for k in range(4):
    bridge.seed_uwbfall_cache(k, "train"); bridge.seed_uwbfall_cache(k, "test")   # reuse srcv features
    bundle = uwbfall.fit(bridge.pipeline_sources(k, "train"), srcv.events(k, "train"), cfg,
                         out_dir=f"/home/user/data_raw/srcv/runs/<name>/fold_{k}")
    preds[k] = uwbfall.predict(bundle, bridge.pipeline_sources(k, "test"))
srcv.score_pooled(preds); srcv.scoring.bootstrap_ci(preds)
```

The same loop is available as a CLI: `python3 -m srcv.run_pipeline <name> [--config cfg.json]`. It writes
`srcv/results/<name>/fold_k.csv` and `summary.json`.

`bridge.pipeline_composition(k, part)` returns the composition DataFrame in the column layout that
`uwbfall.sources_from_composition` reads: `source_t0/source_t1` exact, plus `event_t0, event_t1, fall_type, quality,
needs_review, ignore`.

### Training labels

`srcv.events(k, "train")` follows the published train events:

- `ignore=false` on every row;
- `quality` and `needs_review` are present, so a pipeline can apply its own flagged-fall policy, as the prototype and E4 do.

Post-processing chosen on training OOF should be scored with `srcv.score(df, k, part="train")`. That call applies
`--ignore-flagged` semantics, exactly like the CLI on the published train streams.

### Promotion rule: `srcv.promote(A, B, n_boot=5000, seed=0, val_A=None, val_B=None)`

`A` is the incumbent and `B` the candidate. Each is `{fold: DataFrame | csv}` covering the fold's test streams, or
one DataFrame/CSV covering `cv{k}_test_*`, or a **list** of those, one entry per seed.

For a list, the per-subject TP/FP/FN counts are averaged over seeds, and the per-fold scores are seed means. You can
instead average probabilities before decoding and pass one arm. Seed-average stochastic models (≥3 seeds) before
comparing, as PLAN §2 requires.

1. **Subject attribution** (`cluster_counts`), done once with the official `_match_stream` at θ = 0.1, 0.3, 0.5 and 0.7:
   - TP and FN go to the subject of the ground-truth event's source recording.
   - An FP (an unmatched prediction outside the ±2 s ignore zones) goes to the subject of the composition segment it overlaps most. That is the fall subject for its Chute/Autre/Empty clips, or `ADL_n` for activity pieces.
   - The totals equal the official counts; this is asserted on every call.
2. **Paired, stratified subject-cluster bootstrap.** The 24 fall subjects and the 8 ADL subjects are resampled with replacement, separately, 5000 times, using the same draws for A and B. Ranking F1 = mean over θ ∈ {0.1, 0.3, 0.5} of 2TP / (2TP + FP + FN), computed from the pooled counts. The point estimate equals `score_pooled` exactly.
3. **Decision:** `promote = P(Δ > 0) ≥ 0.90 AND n_folds_improved ≥ 3`. Folds are scored with the official per-fold ranking F1, and an improvement means a strictly greater score. If both `val_A` and `val_B` are given (floats, or val prediction CSVs/DataFrames scored with `score_val`), the decision also requires `val_B ≥ val_A − 0.02`.

The returned dict contains:

- `pooled_A`, `pooled_B`, `delta`, `delta_ci95`, `p_delta_gt_0`
- `per_fold`: A, B and delta per fold
- `n_folds_improved`, `val_A`, `val_B`, `val_ok`, `promote`

CLI: `python3 -m srcv.compare <A> <B> [--val-a X --val-b Y]`. Here `<A>`/`<B>` are directory names under
`srcv/results/`, or a comma list of such names (seeds).

## 5. Baseline: the prototype recipe on SR-CV (`python3 -m srcv.baseline_prototype full`)

The recipe is `analysis/prototype_detector.py` 'full', unchanged:

- the same block features, with the PCA fingerprint fitted on fold-train only;
- HGB with the same parameters and `random_state=0`;
- 4-fold GroupKFold-by-stream OOF on the fold's 13 regenerated train streams;
- the 80-point post-processing grid, chosen on that OOF with the official evaluator.

The final HGB is then fitted on all of fold-train and predicts the fold's held-out test streams.

Prototype 'full' recipe, per fold:

| fold | fold-train OOF (leaky) | post (thr, smooth, min_dur, gap) | **test rank F1** | F1 @0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | med. start err | FA/h |
|---|---|---|---|---|---|---|---|
| 0 | 0.9239 | 0.3, 3, 0.8, 0.6 | **0.8438** | .875/.859/.797/.438 | 55/11/7 | 0.32 s | 5.32 |
| 1 | 0.8901 | 0.5, 5, 0.8, 0.2 | **0.8160** | .880/.832/.736/.400 | 52/11/10 | 0.24 s | 5.36 |
| 2 | 0.8860 | 0.4, 5, 0.8, 0.2 | **0.8667** | .900/.883/.817/.583 | 53/4/10 | 0.22 s | 1.95 |
| 3 (P3_3) | 0.8741 | 0.5, 5, 0.8, 0.6 | **0.7087** | .775/.721/.631/.432 | 40/5/26 | 0.19 s | 2.42 |

Pooled over all 4 folds (20 streams, 8.24 h, 253 evaluable falls):

| system | **pooled SR-CV rank F1** | 95% CI (subject-cluster bootstrap) | F1 @0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | med. start err | FA/h | official val (for reference) |
|---|---|---|---|---|---|---|---|
| prototype recipe (`baseline_prototype`, HGB early_stopping='auto') | **0.8113** | [0.727, 0.869] | .860/.826/.748/.463 | 200/31/53 | 0.22 s | 3.76 | 0.8840 |
| uwbfall HGB only, no replay (`run_pipeline uwb_hgb_noreplay`, early_stopping=False) | **0.8052** | [0.721, 0.865] | .856/.835/.724/.465 | 203/30/50 | 0.28 s | 3.64 | — |
| uwbfall HGB only + annotator replay B (`run_pipeline uwb_hgb_replay`) | **0.8340** | [0.747, 0.895] | .852/.844/.807/.675 | 205/28/48 | **0.04 s** | 3.40 | — |

The uwbfall rows were trained on official train and scored on val with `run_pipeline --val-only`. Val scores for
those runs: HGB without replay **0.8922** (62/7/5 @0.3), HGB + replay **0.9216** (63/6/4 @0.3).

### First live use of the promotion rule: does annotator replay help the HGB detector?

```
python3 -m srcv.compare uwb_hgb_noreplay uwb_hgb_replay      # + val clause via srcv.promote(..., val_A=csv, val_B=csv)
```

| | A = HGB, no replay | B = HGB + replay B |
|---|---|---|
| pooled SR-CV rank F1 | 0.8052 | 0.8340 |
| Δ (95% CI), P(Δ>0) | | **+0.0288** [+0.008, +0.049], **P = 0.996** |
| per-fold Δ | | +0.037 / +0.016 / +0.011 / +0.054 → **4/4 folds improve** |
| official val | 0.8922 | 0.9216 (no drop) |
| **decision** | | **PROMOTE** (`results/promote_hgb_replay_vs_noreplay.json`) |

Replay mainly fixes boundaries:

- F1@0.5 rises from 0.724 to 0.807 and F1@0.7 from 0.465 to 0.675.
- The median start error falls from 0.28 s to 0.04 s.
- Fold 3, the P3_3 fold, goes from 0.702 to 0.756. FN @0.3 there goes from 25 to 23 and FP from 5 to 3.

This confirms E2's val finding (+0.044 on the prototype) on leak-free subject-disjoint data.

The open question in PLAN §0 is not answered by this test. That question concerns replay on top of the TCN or the
blend, which can now be tested the same way.

The mean over folds is 0.809 ± 0.070 (sd), which matches the pooled 0.811. Peak RSS was 0.97 GB and wall time
7.5 min for all 4 folds, with features cached.

### What SR-CV already says

1. **Val is optimistic for this recipe.**
   - The leak-free subject-disjoint estimate is **0.81** (CI 0.73–0.87), against 0.884 on val.
   - Even without the P3_3 cluster, the pooled rank F1 is **0.839** (`pooled_rank_f1_without_Test_P3_3`). Val's 0.884 sits at the top of the per-fold range (0.71–0.87).
   - This agrees with the data team's "plan against ~0.86, val is the most favourable rotation".
2. **D3 blind spot.**
   - When Test_P3_3 (the only D3 subject with evaluable falls) is held out, the prototype detects **0/16** of its falls at any θ. Not a single prediction lies within ±5 s of any of them.
   - Fold 3 drops to 0.709. If a test D3 subject behaves like P3_3, this costs about 0.03 pooled and about 0.12 on that split: fold 3 scores 0.828 without the P3_3 cluster and 0.709 with it.
   - This is the W4 "D3 hedge" risk, now measured.
3. **D2 recall is subject-dependent.** Recall @0.3 per subject:
   - P2_2: 2/6
   - P2_3: 12/18
   - P2_7: 11/16
   - P2_5: 8/11, with 6 FPs
   - every D1 subject: 0.81–1.00.
4. **Leaky fold-train OOF overestimates by 0.03–0.08** (0.874–0.924 vs 0.709–0.867).
5. **FP sources.** Of the 31 FPs @0.3:
   - 6 sit in ADL pieces, from ADL_6 and ADL_12;
   - 25 sit in fall-subject clips (fall-recording margins, Autre/Empty), led by P2_5 (6) and P2_9 (4).


### Current best candidate (uwbfall default = 0.75 TCN 3-seed bag + 0.25 HGB, then replay B) on SR-CV

DEFAULT_RESULT

### Recipes for the next work packages

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
python3 -m srcv.run_pipeline <name> --config my.json              # 4 folds, about 16 min/fold with the TCN bag, 2-3 min HGB-only
python3 -m srcv.run_pipeline <name> --config my.json --val-only   # official train -> val (clause c)
python3 -m srcv.compare <incumbent> <candidate>                   # P(Δ>0), per-fold Δ, decision
```

W2 needs these arms; each is a uwbfall config:

- **TCN alone + replay:** `{"hgb": {"enabled": false}}`
- **blend without replay:** `{"boundary": {"method": "none"}}`

Compare each against `uwb_default`.

## 6. Honest limitations

- **The fold structure is a model of the test, not the test.**
  - The official test streams are written by an unshipped `scripts/rebuild_test.py`, with an unknown seed and possibly unknown details. SR-CV uses the public generator code, `_write_stream` and the configured density.
  - The 24 fall subjects come from the same 3 datasets (rooms and radar placements) as the test subjects will. SR-CV is subject-disjoint, **not** room-disjoint, and neither is the official test, as far as the package shows.
- **Only 8 ADL subjects.**
  - The ADL stratum of the bootstrap has 8 clusters. FP uncertainty from ADL content is therefore estimated coarsely.
  - Each fold's test ADL comes from 2 subjects. With ADL_11 (1153 s) the fold-1 ADL pool is fully used, as in the official val.
- **One regeneration per fold.**
  - The test-stream context (margins, ADL cuts, ordering, crossfade neighbours) is drawn once per fold with seed 7200+k.
  - The E1 bank shows that context noise alone gives about ±0.02 per 2-h split. The pooled 8.2 h reduces this, but it is not zero. A model comparison is paired on the same streams, so this noise mostly cancels in Δ.
- **t_last.** The published plan used a clip end one frame shorter than the aligned segment (V2). SR-CV follows the published behaviour (`(n−1)·0.02`). One val clip (Test_P3_8/Chute02_270) is still 1 frame off, which is immaterial.
- **Training OOF inside a fold is still leaky.** The recipe's GroupKFold-by-stream on regenerated train streams shares ADL files and subjects across streams, exactly as on the official train. This affects only operating-point choices *inside* a fold. The SR-CV test score itself is leak-free: the test subjects, their Autre/Empty files and the test ADL subjects never appear in fold-train.
- **Prototype seed sensitivity.** The HGB uses `early_stopping='auto'` with `random_state=0`, as in the recipe. Model-seed variance (X5: sd about 0.012 on val) is **not** in the baseline CI.
- **Val stays a confirmation set.** Folds 1–3 contain original val subjects (P1_8/9, P2_8/9, P3_8/9, ADL_0/11). SR-CV models are trained on regenerated streams that include *some* val subjects in fold-train. Therefore never tune on val and SR-CV jointly. Report official val only as the third clause of the rule.

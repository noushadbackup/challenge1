# Expert review X5: MLOps and reproducibility

**Verdict: sound with changes.** The modelling plan is reasonable. Three things are missing: a pipeline that the organisers can re-run, a noise-aware promotion rule, and a compute budget that fits 2 cores, 5 GB of RAM and 16 GB of disk.

The challenge README says: *"Top-ranking teams will be required to provide their code and models for verification."* If we win on the leaderboard but cannot reproduce the result from raw test `.h5` files, the win does not count. Section 5 of PLAN.md covers this with one line ("everything scripted, seeded, versioned"). It needs to be a work package with tests.

Check scripts are in `/home/user/data_raw/agents/X5/` (`rss_extract.py`, `cmp.py`, `det.py`, `seeds.py`, `thr.py`, `bench.py`, `regen*.py`). Every number marked *measured* comes from those scripts, run in this session. Numbers marked *estimate* are extrapolated from them.

---

## 1. What I measured

| # | check | result |
|---|---|---|
| M1 | Re-extract `val_00` from raw `.h5` with `analysis/extract_features.py::process` and compare with the cached `feats/val_00.npz` | **All 7 arrays bit-identical** (sha256 prefixes e.g. `prof_02` 43ba11d17ead). The cache is a faithful function of the raw data. |
| M2 | Peak RAM and time of that extraction (one process, 1 thread) | **3.55 GB peak RSS, 26 s per 30-min stream.** The README's `n_jobs=3` would need about 10.6 GB, which breaks our 5 GB cap. Only `n_jobs=1` is safe as written. |
| M3 | float16 fidelity of `prof_02` (x1000) | Quantisation error is 9e-5 of the profile norm and 0.2-0.6 % of the median 1-s profile change. **Not a bottleneck**: static-change and junction features computed from the cache are safe. |
| M4 | Prototype (`full`) refit with `random_state=0`, 2 threads | **Reproduces 0.8840 exactly** (62/6/5, median start error 0.22 s). |
| M5 | Same model, 1 vs 2 OpenMP threads | Probabilities differ by up to 1.7e-5. The submission CSV (rounded to 3 dp) is **identical** (85 events, same hash). There is a small risk that a different thread count flips a threshold, so pin the thread count. |
| M6 | **Hidden randomness in the prototype.** `HistGradientBoostingClassifier` has `early_stopping='auto'`, which switches on above 10 000 rows (we have 111 268). The model is then fitted on a random, *frame-level, non-grouped* 90 % of train, and `random_state` picks which 10 %. | Val ranking F1 over seeds 0-4: **0.884 / 0.887 / 0.904 / 0.872 / 0.895, mean 0.8885, sd 0.012, range 0.032.** With `early_stopping=False` the result is deterministic: 0.8856. Seed noise alone is as large as most of the gains under discussion. For example, the padding gain is +0.010 to +0.020. |
| M7 | Can the public generator rebuild the published streams? | Re-planning with the public seed reproduces the **fall-clip crops exactly (119/119 val `source_t0`)**, but **not the ADL cuts (36/435 pieces)**. It does not reproduce the stream order at either density (56.71 from the public binding density, 57.19 from `streams.yaml`). So the pseudo-test bank can only be checked for *distributional* parity, not for bit parity. |
| M8 | Cost of building one stream in memory with the generator | Plan 20 s, crossfade plus `.h5` write 52 s, 0.9 GB RSS. A written stream takes **400 MB of disk**, so 50 streams would need 20 GB, which exceeds the **16 GB free** (shared). |
| M9 | CPU training cost on 2 threads (`bench.py`) | ResNet-18 (11.2 M params) on 3x64x200 range-time windows: **25.8 ms per window per train step.** A small dilated TCN (5 layers, 64 channels) on 0.2-s block sequences: **0.047 s per batch of 16 x 120 s**, about **0.7 s per epoch over all 8.2 h** (estimate). |
| M10 | Official evaluator call | 0.2-0.3 s per call. A grid search using the official scorer is cheap. |
| M11 | Installed environment | `torchvision`, `timm`, `sktime`, `pytest`, `optuna` and `mlflow` are **not installed**. `onnxruntime` and `yaml` are. Installed versions differ from the package pins: scikit-learn 1.9.1 vs pinned 1.7.2, and pandas 3.0.5. Python is 3.13. |
| M12 | Package hygiene | `scripts/__pycache__` and `src/abc2027/__pycache__` have already been written into the "read-only" package. `scripts/build_streams.py::main()` **overwrites `streams_events.csv`, `streams_composition.csv` and `reports/*` inside the package**, so calling it as-is would corrupt the ground truth. |
| M13 | Code fragility | `prototype_detector.official()` runs the evaluator with `cwd=PKG`, so a relative `--pred` path fails (I hit this). |

---

## 2. Changes, prioritised

### P0 (must do to win)

**P0-1. A verifiable submission package as its own work package (new executor task E6, or the tail of E5).**
- **Layout:** `uwbfall/` is a pip-installable package (`pyproject.toml`) containing:
  - `features.py`, the only feature code, used by both training and inference;
  - `models/`, `postproc.py` and `cli.py`, with commands:
    - `uwbfall extract --streams DIR`
    - `uwbfall train --config configs/final.yaml`
    - `uwbfall predict --streams streams/test --out submission.csv`
  - `configs/*.yaml`, frozen.
  - `artifacts/`, holding the model files and a `MANIFEST.sha256`.
  - `requirements.lock` (`pip freeze` of the exact env, Python 3.13).
  - `README_REPRO.md`.
- **Model formats that survive version drift:**
  - LightGBM `Booster.save_model("m.txt")`;
  - XGBoost `.ubj`/`.json`;
  - CatBoost `.cbm`;
  - torch `state_dict`, plus an ONNX export run through `onnxruntime`, which is installed.
  - **Do not ship sklearn pickles for the final model** (the env already drifts 1.7.2 to 1.9.1). This favours LightGBM over `HistGradientBoosting` for S3-a.
- **Acceptance tests.** Write these as plain `python3 tests/run_all.py`, or install `pytest`:
  1. Cache parity: `extract(raw val_00) == feats/val_00.npz` (already true, M1).
  2. Two `predict` runs give byte-identical CSVs.
  3. `predict` on val, scored by the official evaluator, matches the recorded score to 4 dp.
  4. Submission schema validator:
     - columns `stream_id,t_start,t_end,score[,t_alarm]`;
     - ids match `test_\d\d` and exist;
     - `t_start < t_end`, both inside `[0, T/50]`;
     - no NaN/inf, scores in `[0,1]`;
     - no duplicate rows.
  5. The package checksum manifest is unchanged (see P0-4).
- **Expected gain:** 0 F1, but it is what makes a ranking eligible. Without it a top rank can be voided.
- **Verify:** a fresh shell in a clean venv (`python3 -m venv /tmp/v && pip install -r requirements.lock -e .`) runs `make reproduce-val` and prints 0.8840 for the prototype config (later, the final config) within 10 min.

**P0-2. Remove the hidden randomness and make the promotion rule seed-aware.**
- **Fix the randomness:**
  - In `prototype_detector.py`, set `early_stopping=False`.
  - In the LightGBM successors, set `n_estimators` from grouped CV and use no in-fit random holdout. If early stopping is used, its validation set must be a *grouped* fold (subject plus ADL file, per the data-team verdict), never a random frame split.
  - LightGBM: `deterministic=True, force_row_wise=True, num_threads=2, seed=s, bagging_seed=s, feature_fraction_seed=s`.
  - XGBoost: `tree_method="hist", nthread=2, seed=s`.
  - CatBoost: `random_seed=s, thread_count=2`.
  - torch: `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(2)`, `DataLoader(num_workers=0)`.
  - Export `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONHASHSEED=0` in every entry point.
- **Promotion rule (replaces §2.5):**
  - Every candidate and its parent are run with **at least 5 seeds**, on the same folds and the same pseudo-test streams.
  - The **paired** difference is computed per (seed, fold or stream) unit.
  - A change is promoted only if the paired-bootstrap 90 % CI of the mean difference is above 0. The current "1 sd" rule does not hold up against the measured seed sd of 0.012.
- **Expected gain:** this avoids keeping changes that are noise. With the current rule, seed 2 vs seed 3 of the *same* model would look like a +0.032 "improvement" (M6). It also makes the final model reproducible bit for bit.
- **Verify:** for each promoted change, the run registry (P0-6) holds at least 5 seed rows per arm, plus the CI.

**P0-3. Memory-safe extraction from raw `.h5`, for test inference and for the pseudo-test bank.**
- **Rewrite `extract_features.process`** to work per radar and in time chunks:
  - read `cir[r, i:i+C]` with h5py slicing (the file is chunked at 5000 frames);
  - carry the EMA state (`lfilter` `zi`) and the last frame across chunks for `e_diff`;
  - compute in float32 except the energy sums.
- **Target:** under 1.2 GB peak per process (estimate), which allows 2 parallel jobs within 5 GB. The current version peaks at 3.55 GB (M2).
- **Parity requirement:** chunked output equals the current cache bit for bit, or to within 1e-6 relative for the float32 keys. The `prof_02` float16 values must match exactly. If EMA chunking breaks bit-equality, keep a recorded tolerance and re-score val to show the F1 is unchanged.
- **Version the cache by extractor hash** (`feats/<sha8>/`) so that a feature change can never silently mix with old caches.
- **Expected gain:** 0 F1. It makes P0-1 and P0-4 feasible on this machine, and avoids being OOM-killed by the other agent's RAM use.
- **Verify:** `/usr/bin/time`-style `ru_maxrss` stays under 1.5 GB per stream in the CI log, and the parity test passes on `val_00` and `train_12` (the short stream).

**P0-4. A pseudo-test or regeneration harness that cannot damage the package or the disk (constrains E1).**
- **Never call `build_streams.main()`.** It rewrites the package CSVs (M12). Import only:
  - `abc2027.streams.prepare_split(..., rng=np.random.default_rng(seed))` and `finish_plan`;
  - a copy of the crossfade loop from `_write_stream` that returns `cir` in memory.
- **Pipe the output straight into the P0-3 extractor** and save only the feature `.npz` (about 35 MB per stream) plus the events and composition CSVs into `/home/user/data_raw/regen/<split>_s<seed>_<i>`. Write no `.h5`: 50 streams would take 20 GB (M8).
- **Package hygiene:**
  - run everything with `PYTHONDONTWRITEBYTECODE=1` (or `python3 -B`);
  - record now a `sha256` manifest of every package file except `segments/` and `streams/`, which are size and mtime checked;
  - check the manifest in the acceptance tests.
- **Parity checks.** Bit parity is impossible (M7), so check distributions instead:
  - falls/hour;
  - pre and post margin quantiles;
  - ADL chunk length quantiles;
  - number of junctions;
  - fall-to-fall adjacency rate.
  - Each must fall within the range seen across the 5 published val streams. KS tests on margins must give p > 0.05.
- **Use a fresh seed per stream index:** `SeedSequence(seed).spawn`.
- **Cost (estimate, from M2+M8):** about 100 s per stream single-process, so 50 streams take about 85 min and 1.8 GB of disk.
- **Expected gain:** a usable variance-aware selection set within budget. Its known limit is that it reuses the same 119 val falls (X3's point), so it measures context variance, not subject variance. Also report leave-subject-out CV as X3 proposes.
- **Verify:** the manifest is unchanged after a 50-stream run; `du -sh regen/` is under 2 GB; the parity table is in `reports/team/E1_parity.md`.

**P0-5. Re-scope the deep-model stages to the compute budget (§3 S3-c/S3-d, §4 P4).**
- **ResNet-18 is out of budget.** At 25.8 ms per window per step on 2 threads (M9), training on 4-s windows at a 1-s hop over train (about 22 k windows) costs about 10 min per epoch. With the plan's 5 folds x 3 seeds x about 20 epochs that is about 50 h of 2-core time (estimate). At the 0.2-s hop needed for localisation it is about 5x more.
- **Other blockers:** `torchvision`/`timm` are not installed, and ImageNet weights raise a rules question (see P1-2). ActionFormer and TriDet also need their repos installed, and their CUDA-oriented NMS ops need CPU fallbacks.
- **Replace them on the critical path with:**
  - (a) LightGBM (S3-a);
  - (b) a small dilated TCN or MS-TCN++-style model on the cached 0.2-s block features. It costs about 0.7 s per epoch (estimate from M9), so the full 5x3 bagging grid takes minutes. Reference: https://github.com/sj-li/MS-TCN2 (MIT);
  - (c) a GRU or CRNN head on the same inputs.
- **Keep ActionFormer as P2:** https://github.com/happyharrycn/actionformer_release (MIT). Run it only on TCN embeddings, with a single seed, as a probe.
- **MiniRocket (S3-f):** `sktime` is not installed. Use `aeon` (https://github.com/aeon-toolkit/aeon, BSD-3) if wanted. It is P2.
- **Expected gain:** frees about 50 h of CPU for the boundary work (X1/X2 P0: annotator replay), where the measured points are.
- **Verify:** a compute ledger in the run registry (wall time and peak RSS per run). Every P4 model must log its train cost before it is promoted.

**P0-6. Run registry and a pre-registered final selection.**
- **Registry contents.** No MLflow is installed; plain files are enough and audit-friendly. Each run writes `runs/<run_id>/` with:
  - `config.yaml`;
  - `git rev-parse HEAD` plus `git diff` (if dirty);
  - feature-cache hash;
  - `env.txt` (pip freeze, thread env);
  - OOF and val prediction parquet;
  - official evaluator JSON for train-OOF, val and pseudo-test;
  - wall time and peak RSS.
  - Append a row to `runs/index.csv`.
  - MLflow (https://github.com/mlflow/mlflow, Apache-2.0) is optional if it can be pip-installed.
- **Final selection rule, written now:** the final config is the best *seed-averaged* grouped-CV plus pseudo-test candidate under the P0-2 rule. Val is a go/no-go check only.
- **Final fit:** train+val with fixed `n_estimators` and epochs taken from CV, bagged over 5 seeds. Score averaging across seeds also removes the M6 noise from the submitted model.
- **Expected gain:** cuts the risk of picking a val-lucky config (5 streams, 67 falls, CI about +/-0.06), and gives the verification audit trail.
- **Verify:** `submission.csv` traces to exactly one `run_id`, whose config hash matches `configs/final.yaml`.

### P1

**P1-1. One-command reproduction of every claim in REPORT.md.**
- Add a `Makefile` (or `run_all.sh`) with targets `features`, `prototype`, `figures`, `reproduce-val`, `test`.
- `prototype` must use absolute paths in `official()` (M13). Better, wrap the evaluator in a helper that resolves paths and returns the JSON.
- **Verify:** `make prototype` reprints `summary.json` exactly.

**P1-2. A rule-compliance ledger, kept in the repo for the verification step.** One row per data source and per trick, with a yes/no ruling from the team lead:
- `streams/`, `segments/` and regenerated streams (package data, allowed);
- pretrained weights, if any. ImageNet weights are external data, so they need an explicit ruling. Prefer training from scratch;
- generator-artefact features: junctions, the spacing-3 repeat share, hard cuts;
- **transductive cross-stream ADL duplicate masking on test.** This has the highest verification risk. Keep it behind a config switch, default off, and keep a recorded ablation so that we can submit without it if it is ruled out.
- **Expected gain:** protects the ranking at review time.
- **Verify:** the ledger exists, and every switch has an ablation `run_id`.

**P1-3. Pre-submission sanity monitors on the hidden test features.** These need no labels.
- Predicted events per hour vs the expected range. Expect about 58 falls/h, of which about 75-85 % are D1/D2-evaluable plus D3. Flag anything outside 35-75 /h.
- Per-radar feature drift:
  - PSI of `diff_max`, `static*`, `jump` against train;
  - adversarial-validation AUC (LightGBM, train blocks vs test blocks). Flag anything above 0.8.
- Fraction of predictions starting within 1 s of a detected junction.
- Detected junction count per hour vs train/val (about 150 /h, to be measured by E1).
- **Expected gain:** catches a changed test generator (plan risk 1) before the one-shot submission, and can trigger the physics-only hedge member.
- **Verify:** the monitor script runs on val (it must pass) and on a deliberately corrupted stream (it must fail).

**P1-4. Compute and RAM budget enforced in code.**
- Every entry point calls `resource.setrlimit(RLIMIT_AS, 5 GB)`, or checks `ru_maxrss` and aborts at 4.5 GB.
- Worker pools are capped at 2.
- Temporary files go to scratch and are cleaned up on exit.
- **Verify:** registry rows show `peak_rss_GB < 5`.

**P1-5. Fold and seed hygiene.**
- **Grouping:** fold assignment is a saved artefact (`folds.json`) grouped by subject *and* ADL file, as the data verdict requires. The prototype uses `GroupKFold(4)` by stream, which the data team showed is leaky.
- **One fold file for every model,** so OOF predictions can be stacked (S5) without leakage.
- **Expected gain:** honest OOF for the post-processing and stacking stages. The train-OOF numbers used for tuning are currently optimistic.
- **Verify:** an assertion that no subject and no ADL file appears in both the train and validation parts of any fold.

### P2

- **P2-1. ONNX inference graph.** Export the torch models (`torch.onnx.export`) and run them with `onnxruntime` CPU, so verifiers need neither torch nor matching CPU kernels. Check parity against torch to 1e-5.
- **P2-2. Container.** Add a `Dockerfile` (`python:3.13-slim` plus `requirements.lock`). Verifiers can then run `docker run ... uwbfall predict`. It is optional, because the lock file plus the CLI already meets "provide code and models".
- **P2-3. Speed.** Inference from raw takes about 26 s per stream at present (M2), so the 5 test streams take about 2-3 min. No optimisation is needed. Note it in `README_REPRO.md` as a test.

---

## 3. Compute budget (2 threads, 5 GB RAM, about 16 GB disk)

| item | cost | source |
|---|---|---|
| raw to feature cache, one 30-min stream | 26 s, 3.55 GB RSS (current) | measured M2 |
| feature blocks for all 18 cached streams | 15 s | measured |
| HGB/LightGBM-size fit, 111 k rows x 64 features | 7 s (2 threads) | measured M4 |
| official evaluator call | 0.2-0.3 s | measured M10 |
| pseudo-test stream (plan + crossfade + features) | about 100 s, about 35 MB disk if no `.h5` is written | estimate from M2, M8 |
| 50-stream pseudo-test bank | about 85 min, about 1.8 GB | estimate |
| TCN epoch over all streams | about 0.7 s | estimate from M9 |
| ResNet-18, 4 s @ 1 s hop, one epoch over train | about 10 min | estimate from M9 |
| plan's 5 folds x 3 seeds x 20 epochs, ResNet-18 | about 50 h | estimate; do not run |

---

## 4. Plan text to amend

- **§2.5:** replace the promotion rule with the P0-2 version (at least 5 seeds, paired bootstrap CI above 0).
- **§2.2:** add "in-memory generation, features only, never `build_streams.main()`, package manifest check" (P0-4).
- **§3 table:** move S3-c (ActionFormer/TriDet) and S3-d (ResNet-18) to optional P2 probes. Add the TCN/CRNN on block features as the deep member.
- **§4 P6:** split it into P0-1 (package plus tests, built *in parallel from now on*, not at the end) and P0-6 (frozen selection, final fit with fixed iterations, seed bagging).
- **§5 risks:** add version drift, OOM from the co-tenant agent, disk exhaustion, package corruption, the transductive-trick ruling, and the hidden early-stopping randomness.

## 5. Metrics to add (all logged per run in the registry)

- Seed sd and fold sd of ranking F1. Paired-difference mean and 90 % bootstrap CI against the parent.
- Wall time and peak RSS per stage. Disk used by caches.
- Feature-cache parity hash (raw to features) and submission hash (determinism).
- Predicted events/hour on each split and on test.
- Adversarial-validation AUC and PSI between train and test features.
- Share of predictions near junctions, and share inside ignore zones (+/-2 s), on val and pseudo-test.
- An official-evaluator JSON for every run, not only for the final run.

## 6. Risks

1. **Verification failure:** code that only runs from our cached `.npz` files, sklearn pickles that do not load under the organisers' pinned sklearn 1.7.2, or a missing lock file.
2. **Selection on noise:** seed sd 0.012 and val CI about +/-0.06. Without P0-2 we will promote noise.
3. **OOM or disk exhaustion** on the shared machine. Current extraction uses 3.55 GB per process, and writing `.h5` for 50 regenerated streams needs 20 GB.
4. **Corrupting the package's ground truth** by running `build_streams.py` as a script.
5. **Rules risk** from transductive and generator-artefact tricks, or from pretrained external weights. These need a lead ruling, a switch and an ablation.
6. **Compute sink:** the deep image models and TAL repos on CPU can eat days with an unproven gain. Gate them behind the cheap TCN result.

## References

- scikit-learn `HistGradientBoostingClassifier` (`early_stopping='auto'` is on when n_samples > 10 000): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html (I confirmed the behaviour empirically, M6).
- LightGBM `deterministic` / `force_row_wise`: https://lightgbm.readthedocs.io/en/latest/Parameters.html
- PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- MS-TCN++: https://github.com/sj-li/MS-TCN2 · ActionFormer: https://github.com/happyharrycn/actionformer_release · aeon (MiniRocket): https://github.com/aeon-toolkit/aeon · MLflow: https://github.com/mlflow/mlflow · ONNX Runtime: https://github.com/microsoft/onnxruntime
- These URLs are cited from memory. I did not re-fetch them in this session.

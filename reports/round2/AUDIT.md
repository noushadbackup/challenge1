# Round-2 reproducibility and compliance audit

I acted as a challenge organiser checking a top team's code. I worked from a **fresh clone**, with fresh caches, and ran every number below in this session. Scores come from the official evaluator (`abc2027.evaluate`, `--ignore-flagged`).

- Audited state: commit `b3716b8` plus the integrator's uncommitted working tree, i.e. what the lead is about to commit.
- Evidence: logs and CSVs are in `/home/user/data_raw/audit_out/`. Measurement JSONs and the scripts I used (`meas.py`, `val_csv.py`, `lane1.sh`, `lane2.sh`) are copied to `reports/round2/audit/`.

## Verdict

| # | item | result |
|---|---|---|
| 1a | Fresh clone, install, cold feature extraction | **PASS**. All 18 feature files are byte-identical to the team cache. |
| 1b | `make reproduce-val` from scratch (train13 refit, nothing reused) | **PASS**. The val CSV is **byte-identical** to I1 G7N4 bag 0. Official score **0.9524**, 64/2/3 @0.3. |
| 1c | `make train-final` from scratch (train+val) | **PASS** for the models: 9/10 manifest files are bit-identical and the predictions are identical. **FAIL** for the tooling: `make verify-models` rejects the refit bundle. See fix F2. |
| 1d | Makefile and README usable from a fresh clone | **PASS with fixes**. The Makefile silently runs the code in `/home/user/challenge1` unless `REPO=` is given (fix F1). `pyproject` lacks `lightgbm` (F4). The root README is empty (F5). |
| 2 | Determinism: repeated runs, cold vs warm cache, thread counts | **PASS**. Runs at runtime.threads 2 and 4, and with the OMP env set to 4, give identical CSVs. At 1 thread, 2 of 119 scores move by 0.001 and the official score is identical. |
| 3 | `validate_predictions` on the dry-run CSV | **PASS**: 119 rows, 5/5 streams, exact header. |
| 4 | Rule compliance | **PASS**, with one minor clean-up (package `__pycache__` from round 1) and one disclosed interpretation risk (exploiting the stream generator's structure). |
| 5 | RAM and time of predict on 5 streams (2.05 h) | **PASS**. Cold: 143 s, 1.6 GB peak RSS. Warm: 62–69 s, 1.1 GB. |

**Overall: PASS.** The submission is reproducible bit-for-bit from a clean clone. The remaining items are tooling and documentation fixes (F1–F8) that the lead should apply before or with the commit. None of them changes a prediction.

## 1. Fresh clone, install, documented commands

**Setup**
- `git clone /home/user/challenge1 /home/user/data_raw/audit_clone`.
- No `rsync` is installed, so I overlaid the 82 modified or untracked, non-ignored files with `git ls-files -m -o --exclude-standard | tar`. A `diff -rq` against the working tree, excluding ignored caches, was empty.
- The clone is 64 MB. It contains `models/final/` (committed, 3.7 MB).
- `models/fallback_physics/` is **untracked**. It must be committed (F6).

**Install per the README**
- `python3 -m venv --system-site-packages /home/user/data_raw/audit_venv`, then `pip install --no-build-isolation --no-deps ./uwbfall`. Both worked.
- The installed `uwbfall predict` entry point, run from `/tmp` without `PYTHONPATH`, gave the reference CSV byte for byte (sha256 `754f6a3c…`).
- `--no-deps` hides one problem: `uwbfall/pyproject.toml` does not list `lightgbm` (used by the clip stage) or `threadpoolctl`. On a truly empty environment, `pip install ./uwbfall` would install a package that fails at fit and predict time (F4).
- Versions used: Python 3.13.16, torch 2.14.1+cpu, scikit-learn 1.9.1, numpy 2.5.3, lightgbm 4.7.0, pandas 3.0.5, scipy 1.18.1, h5py 3.16.0.

**Cold feature extraction**
- I ran `make features` with a fresh `UWBFALL_CACHE=/home/user/data_raw/audit_cache`: 496 s, 1.37 GB peak RSS.
- All **18/18** `feat_*.npz` files are **byte-identical** to the team cache `/home/user/data_raw/uwbfall_cache`.

**`make reproduce-val`** (`REPO=<clone> MODELS=/home/user/data_raw/audit_models`, so no prebuilt bundle was reused)
- The train13 bundle was refitted from scratch: 1035 s for the whole target, 1.45 GB peak.
- Against the team's `srcv/runs/final_b0/official_train`, every model and OOF file is identical (`hgb.pkl`, `tcn_{0,1,2}.pt`, `tcn_norm.npz`, `pca.pkl`, `clip.pkl`, `drift_ref.json`, `oof*.{npz,csv}`, `config.json`). `meta.json` differs only in `timings.*` and `peak_rss_gb`.
- The `cmp` against `work/I1_clip/preds/uwb_default/G7N4/val.csv` reported **byte-identical**.
- Official CLI, run directly: `PYTHONPATH=src python3 -m abc2027.evaluate --gt streams_events.csv --pred …/val_final_train13.csv --streams streams/val --ignore-flagged` gives **f1_mean 0.95238**.
  - TP @0.1/0.3/0.5/0.7 = 65/64/61/53.
  - 64/2/3 @0.3.
  - Median start error 0.055 s; 0.98 FA/h.
  - This matches FINAL.md exactly.

**`make train-final`** (train+val, from scratch, 1087 s, 1.55 GB peak)
- 9/10 files of `models/final/MANIFEST.sha256` are OK. `meta.json` FAILED, differing only in `git_rev` (`b3716b8` vs `31f8671`, the rev the shipped bundle was trained at), `timings.*` and `peak_rss_gb`.
- `oof.npz`, `oof_final.csv` and `oof_candidates.csv` are also identical.
- Predicting the fake test with the refit bundle gives the **same CSV** (sha256 `754f6a3c…`).
- Consequence: the uwbfall code changed between `31f8671` and `HEAD`, but those changes do not affect training or prediction.
- **Defect.** SUBMISSION.md §2 says that if the bundle is missing, you should run `make train-final` and then `make predict-test`. After a refit, the `verify-models` prerequisite exits with status 2, so `predict-test` aborts. See F2.

**Other documented targets**, run from the clone
- `make checks`: configs OK; the drift-monitor sanity test is OK (clean val gives no ALERT; ×10, no-repeat and no-junction drifts all alarm). 35 s.
- `make predict-test TEST_DIR=/home/user/data_raw/fake_test/streams/test SUB=…/audit_out/sub_dry`:
  - verify-models OK;
  - 119 rows, 0 drift ALERTs, `SUBMISSION CHECK: OK`;
  - CSV byte-identical to the integrator's `fake_test/submission/submission.csv` and `submission_split_test.csv`.
- `make predict-test-fallback` (with `FALLBACK=<clone>/models/fallback_physics`): verify-fallback OK, 116 rows, `SUBMISSION CHECK: OK`, byte-identical to the integrator's `submission_fallback.csv`.
- I did not re-run `make srcv` (75–110 min). Its fold and val CSVs were already shown byte-identical to I1 by the integrator, and the reproduce-val refit above covers the same code path.
- I did not run `make dry-run` itself, because it hard-codes `SUB=/home/user/data_raw/fake_test/submission` and would overwrite the integrator's evidence (F3). The equivalent `predict-test` on the same fixture is reported above.

## 2. Determinism

All runs predict the 5 fake-test streams (val copied as test_00..04) with the shipped FINAL bundle, using the clone's code.

| run | cache | threads | wall | peak RSS | CSV sha256 |
|---|---|---|---|---|---|
| t2_cold | fresh, page cache dropped | runtime.threads=2 | 143.2 s | 1.54 GB | `754f6a3c…` |
| t2_warm | warm | 2 | 69.2 s | 1.08 GB | `754f6a3c…` |
| t2 with `OMP_NUM_THREADS=4` in env | warm | 2 (config wins) | 86.3 s | 1.08 GB | `754f6a3c…` |
| repo copy `models/final` | warm | 2 | 62.0 s | 1.10 GB | `754f6a3c…` |
| installed venv entry point | warm | 2 | 48.7 s (internal) | – | `754f6a3c…` |
| refit bundle (§1) | warm | 2 | – | – | `754f6a3c…` |
| t4_cold (bundle copy with `runtime.threads=4`) | fresh | 4 | 169.5 s | 1.53 GB | `754f6a3c…` |
| t1_cold (bundle copy with `runtime.threads=1`) | fresh | 1 | 139.9 s | 1.52 GB | `f53f5150…` |

- **Features are thread-invariant.** The `feat_*.npz` files from the 1-, 2- and 4-thread cold runs are byte-identical.
- **1-thread run.** It has the same 119 rows and identical `t_start`, `t_end` and `t_alarm`. 2 rows have a score that differs by 0.001 (TCN float summation order, after rounding to 3 decimals). Scored against the fake ground truth, both CSVs give exactly the same ranking result (in-sample f1_mean 0.97015, start error 0.050 s, FA/h 0.488). This is not a performance estimate, since the model was trained on these streams.
- **The thread knob is `runtime.threads` in the bundle's `config.json`.** It is enforced by `torch.set_num_threads` and `threadpoolctl` inside predict. `OMP_NUM_THREADS` in the environment does not change the result, which is good, but SUBMISSION.md presents the env var as what matters (F7).
- **No cross-stream dependence.** Predicting test_02 alone in its own folder gives exactly the rows it gets in the 5-stream batch. So does the same file renamed `stream_00.h5`, i.e. id test_00. The output does not depend on the other streams or on the id.

## 3. `validate_predictions` on the dry-run CSV

I ran `reports/round2/audit/val_csv.py`, which calls the official `abc2027.evaluate.validate_predictions` with the official `stream_durations` of the folder. It passed on `fake_test/submission/submission.csv` and on my own clone-produced CSV:
- 119 records; header exactly `stream_id,t_start,t_end,score,t_alarm`; LF line endings;
- 5/5 streams present (34/31/31/18/5 rows); no duplicate rows;
- scores 0.574–0.996;
- every `t_alarm` lies in [t_start, t_end];
- durations 1.0–4.4 s.

## 4. Rule compliance

| check | how | result |
|---|---|---|
| Package unmodified | Recomputed `e1lib.package_manifest()` before and after all audit runs | `fb0e24027626640206472c8663f6bc1ab6aa1010e430fce42396ac7957b41aaf` both times, equal to `work/E1_pseudo_test/manifest_before.txt`. No package file is newer than 2026-10-13 21:17, so nothing was written during round 2. |
| Package `__pycache__` | `find` | `src/abc2027/__pycache__/` and `scripts/__pycache__/` were created 2026-10-13 20:38–21:16 in round 1, before the `PYTHONDONTWRITEBYTECODE` rule. They are generated bytecode and source files are untouched, but strictly they are writes into the package. Minor (F8). |
| No external or LIARA data, no pretrained weights | grep over `uwbfall srcv analysis work Makefile` for `liara`, URLs, `urllib`, `requests`, `wget`, `curl`, `download`, `pretrained`, `torch.hub`, `torchvision`, `huggingface`. I also inventoried every absolute path in the code. | No hits. All paths point to the package, the repo or `/home/user/data_raw` team caches. All models are trained from scratch: HGB, a TCN with random initialisation and fixed seeds, LightGBM and PCA. |
| Libraries open source | import inventory | numpy, scipy, pandas, scikit-learn, torch, h5py, lightgbm, threadpoolctl, plus the package's own `abc2027`. |
| Transductive cross-stream ADL mask OFF | `final.json`, the deployed bundle's `config.json`, `DEFAULT_CONFIG` | `adl_duplicate_mask: false` everywhere. It is also listed in `NOT_IMPLEMENTED`, so `check_implemented` raises if anyone sets it to true. |
| No cross-stream information at predict time | code read of `pipeline.predict` (loop over streams; features, models, replay and NMS are per stream; PCA and TCN norm are fitted on training only), plus the empirical invariance test in §2 | PASS. The drift monitors pool over test streams, but they only print and write JSON. They never change predictions, and switching to the fallback is a human decision. |
| No test labels | none exist; predict reads only h5 (`composition_of` and the events CSV are used only in fit, retune and drift reference) | PASS |
| Val labels | The deployed bundle is trained on train+val labels. That is allowed: val is labelled training data in the package. Its operating point comes from its own 4-fold training OOF (`meta.postproc = [0.4, 3, 0.8, 0.6]`). Round-2 choices (gate 0.3, floor 0.7, NMS 4 s, segclf feature set) were made on train OOF or SR-CV, per R2_EVAL and I1. | PASS for the rules. Process note: in round 1, val was scored for a few frozen candidates (E2: 2 candidates; E4: the w = 0.5 and 0.75 blends, with w chosen on OOF), and C3 → FINAL was accepted despite val 0.9646 → 0.9524. That is mild val exposure but no tuning on val. |
| Use of `segments/` and the generator | SR-CV regenerates streams from `segments/` in memory | Allowed under the task brief: package data. The deployed bundle is trained only on the 18 published h5 streams plus `streams_composition.csv` labels for the segment classifier. |

**Disclosed interpretation risk (not a FAIL).** Several features exploit the published generator's structure:
- junction statistics at the 0.2 s crossfades;
- sample-and-hold repeat shares;
- the segment classifier that gates on P(fall clip).

The package README describes that construction openly, and the rules forbid only the LIARA source recordings. I recommend that the lead discloses it in the method description. The physics-only fallback bundle exists if the official test writer differs.

## 5. Predict RAM and time (5 streams, 2.05 h of radar, 4-CPU shared VM)

| condition | wall | peak RSS | note |
|---|---|---|---|
| cold feature cache and dropped page cache, 2 threads | **143 s** | **1.54 GB** (internal log 1.61) | about 28 s per 30-min stream, with `make features` running in parallel on the other 2 CPUs |
| warm cache, 2 threads | 62–69 s | 1.08–1.10 GB | |
| cold, 1 thread | 140 s | 1.52 GB | I/O-bound extraction |
| cold, 4 threads | 170 s | 1.53 GB | slower because of CPU contention with the parallel training lane |

Everything is far below the 5 GB per-process budget. The integrator's 260 s cold figure was not reproduced (143 s here); the difference is probably machine load. FINAL.md and SUBMISSION.md should quote a range of 2.5–4.5 min cold.

Training, for reference: `train-final` took 1087 s at 1.55 GB, and the train13 refit 1035 s at 1.45 GB.

## Fixes needed

None of these changes a prediction. F1, F2, F4 and F6 should be done with the commit.

- **F1 — Makefile `REPO` is hard-coded** (`REPO ?= /home/user/challenge1`). From any other clone, `make` silently imports and runs the original repo's code and reads its manifests.
  - Replace with `REPO ?= $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))`.
  - Do the same in `uwbfall/Makefile`, with `/..`.
- **F2 — `verify-models` fails on a legitimately refit bundle.** `meta.json`, with its timings and git rev, is in `MANIFEST.sha256`. Either:
  - drop `meta.json` from `models/final/MANIFEST.sha256` and `models/fallback_physics/MANIFEST.sha256` (re-run `work/FINAL/ship.py` with meta excluded) and add a separate check that `meta.postproc` and `config_hash` match; or
  - make SUBMISSION.md say to use `make predict-test FINAL=<refit> -o verify-models` after a refit.
- **F3 — `dry-run` ignores `FAKE`.** It hard-codes `TEST_DIR=/home/user/data_raw/fake_test/streams/test SUB=/home/user/data_raw/fake_test/submission`. Use `$(FAKE)/streams/test` and `SUB ?=`-style overrides, so that an audit does not overwrite the evidence.
- **F4 — `uwbfall/pyproject.toml` dependencies.**
  - Add `lightgbm` and `threadpoolctl`.
  - Add a pinned lock, e.g. `requirements-lock.txt` with torch==2.14.1+cpu, scikit-learn==1.9.1, numpy==2.5.3, lightgbm==4.7.0, pandas==3.0.5, scipy==1.18.1, h5py==3.16.0. `hgb.pkl` and `clip.pkl` are sklearn and LightGBM pickles that need matching versions.
- **F5 — the root `README.md` contains only `# challenge1`.** Add a short pointer to `SUBMISSION.md` (release day), `uwbfall/README.md` (install/API), `reports/round2/FINAL.md` (evidence) and this audit.
- **F6 — commit the untracked artefacts the runbook depends on:**
  - `models/fallback_physics/` (needed by `verify-fallback` and `predict-test-fallback`);
  - the modified `Makefile` and `SUBMISSION.md`;
  - `work/FINAL/`, `srcv/results/final_b0/*`, `reports/round2/FINAL.md` and this `AUDIT.md` with `reports/round2/audit/`.
- **F7 — SUBMISSION.md wording.**
  - Determinism is governed by the bundle's `runtime.threads` (2), not by `OMP_NUM_THREADS`. Say that 1 thread changes at most 0.001 in a few scores, and that 4 threads was identical in this audit.
  - Update the cold-predict time to "about 2.5–4.5 min".
- **F8 — optional clean-up.** Delete `<package>/src/abc2027/__pycache__` and `<package>/scripts/__pycache__` (round-1 bytecode) so that the package is pristine. Keep `PYTHONDONTWRITEBYTECODE=1`, which the Makefile already sets.

## Audit artefacts (outside git)

- `/home/user/data_raw/audit_clone`: the clone plus overlay, 67 MB, with `uwbfall/build/` from pip install (ignored).
- `/home/user/data_raw/audit_venv`: 14 MB.
- `/home/user/data_raw/audit_cache`: fresh 18-stream feature cache, 457 MB. It is byte-identical to the team cache and can be deleted.
- `/home/user/data_raw/audit_cache_p2`: fake-test cache, 114 MB.
- `/home/user/data_raw/audit_models/{final,final_train13}`: the from-scratch refits, 12 MB.
- `/home/user/data_raw/audit_out/`: all CSVs, logs and official score JSONs.
- To reclaim about 0.6 GB: `rm -rf /home/user/data_raw/audit_{cache,cache_p2,venv}`.

# Round-2 reproducibility and compliance audit

I acted as a challenge organiser checking a top team's code. Every check below was run from a **fresh `git clone`**, outside the team's working tree. Scores come from the official evaluator (`abc2027.evaluate`, `--ignore-flagged`).

The audit was done in two passes:

- **Pass 1 (earlier session, cut off before sign-off).** It audited `b3716b8` plus the integrator's uncommitted tree, overlaid on a clone at `/home/user/data_raw/audit_clone`. Its logs are in `/home/user/data_raw/audit_out/` and its scripts in `reports/round2/audit/`.
- **Pass 2 (this session).** It re-audits the **committed** state.
  - The lead committed pass 1's tree as `31d30ed`. Against the pass-1 overlay, no shared file differs; the commit only adds REVIEW and AUDIT files. The lead's WIP commit `0f2949b` then picked up this pass's Makefile and `verify_bundle.py` fixes.
  - The clone is `/home/user/data_raw/audit_clone_head`. Logs are in `/home/user/data_raw/audit_out/head/`; measurements and scripts are in `reports/round2/audit/head/`.
  - Unless marked "pass 1", **every number below was produced in pass 2.**

## Verdict

| # | item | result |
|---|---|---|
| 1a | Fresh clone and install | **PASS**. `git clone` gives a complete tree (69 MB, models included). `pip install ./uwbfall` now declares every runtime dependency, and the lock file matches the environment exactly (F4 fixed). |
| 1b | `make reproduce-val` from the clone | **PASS**. The val CSV is **byte-identical** to the reported one (sha256 `d9f3ec22…`). The official CLI gives **0.95238**, 64/2/3 @0.3. |
| 1c | `make train-final` from scratch (train+val) | **PASS**. 9/10 files are bit-identical, and so are `oof*`. The predictions are byte-identical. The new `verify-models` accepts the refit (F2 fixed). |
| 1d | Makefile and docs usable from a fresh clone | **PASS after fixes**. F1 (REPO), F3 (dry-run evidence), F5 (README) and F7 (SUBMISSION wording) are fixed and verified. F8 is an optional clean-up that agents must not do. |
| 2 | Determinism: repeat runs, cold vs warm, thread counts, cross-stream | **PASS**. Cold, warm, repo-copy bundle, data_raw bundle, installed entry point, refit bundle, `runtime.threads` 4 and env `OMP_NUM_THREADS=1` all give **one CSV** (`754f6a3c…`). `runtime.threads=1` gives identical rows and times, with 2/119 scores ±0.001. Single-stream and renamed-stream runs give identical rows. |
| 3 | `validate_predictions` on the dry-run CSV | **PASS**: 119 rows, 5/5 streams, exact header, `SUBMISSION CHECK: OK`. |
| 4 | Rule compliance | **PASS**. The package manifest equals `fb0e2402…` before and after. No external data, network or pretrained weights. The transductive mask is OFF and guarded. No test labels exist or are read. Val labels are used only as training data in the deployed bundle. One interpretation risk is disclosed (generator structure). |
| 5 | Predict RAM and time, 5 streams (2.05 h of radar) | **PASS**. Cold: **159 s, 1.56 GB** peak RSS. Warm: **52–61 s, 1.07–1.10 GB**. Both are far inside the 5 GB budget. |

**Overall: PASS.** The committed submission is reproducible bit for bit from a clean clone, and it complies with the rules as written. The fixes this audit required are applied (§6). Four of them are still uncommitted for the lead to commit: README, SUBMISSION, pyproject and `requirements-lock.txt`.

## 1. Fresh clone, install, documented commands

**Clone.** I ran `git clone /home/user/challenge1 /home/user/data_raw/audit_clone_head` at `31d30ed`.
- The clone had a clean status and was 69 MB.
- It contains `models/final/` (3.7 MB) and `models/fallback_physics/` (2.65 MB), each with `MANIFEST.sha256`.
- No overlay was needed; the pass-1 F6 item is resolved by the commit.
- Pass 2's fixes were then copied into the clone, and `cmp` confirmed they equal the repo files.

**Install** (README route)
- `python3 -m venv --system-site-packages /home/user/data_raw/audit_venv_head`, then `pip install --no-build-isolation <clone>/uwbfall`, this time **with** dependency resolution, i.e. without `--no-deps`.
- `pip check`: no broken requirements.
- `pip install --dry-run -r requirements-lock.txt`: nothing would be installed, so the lock equals the audited environment.
- The installed `uwbfall` entry point, run from `/tmp` with `PYTHONPATH` unset, imports `site-packages/uwbfall` and gives the reference dry-run CSV byte for byte (`754f6a3c…`).
- Versions: Python 3.13.16, torch 2.14.1+cpu, scikit-learn 1.9.1, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, lightgbm 4.7.0, h5py 3.16.0, threadpoolctl 3.7.0, joblib 1.6.0.

**`make -f <clone>/Makefile reproduce-val MODELS=/home/user/data_raw/audit_models SUB=…/audit_out/head`**, run from `/tmp`
- `REPO` resolves to the clone (F1).
- The train13 bundle used is pass 1's **from-scratch** refit, not the team's. I re-checked it this session: all 12 model and OOF files are byte-identical to the team's clean-run bundle `srcv/runs/final_b0/official_train`. Only `meta.json` and `train.log` differ, in timings.
- The target's `cmp` printed "val CSV byte-identical to the I1 G7N4 bag-0 result". sha256 is `d9f3ec2296e2fffb…` for both. 2 min 03 s.
- **Official CLI, literal:** `cd <package> && PYTHONPATH=src python3 -m abc2027.evaluate --gt streams_events.csv --pred …/head/val_final_train13.csv --streams streams/val --ignore-flagged`
  - **f1_mean 0.952381**;
  - @0.3: TP/FP/FN 64/2/3, F1 0.962;
  - @0.1: 65/1/2;
  - median start error 0.055 s; FA/h 0.976.
  - Identical to FINAL.md. JSON: `reports/round2/audit/head/val_official.json`.

**`make -f <clone>/Makefile train-final FINAL=/home/user/data_raw/audit_models_head/final`** (from scratch, train+val, 2 threads)
- **1312 s, 1.58 GB** peak RSS.
- The bundle's own training-OOF final rank F1 is **0.9389** (239/10/14 @0.3), equal to FINAL.md.
- Plain `sha256sum -c models/final/MANIFEST.sha256` passes 9/10 files. Only `meta.json` differs, in `git_rev`, `peak_rss_gb` and `timings`. `oof.npz`, `oof_final.csv` and `oof_candidates.csv` are byte-identical to the deployed bundle.
- The new `make verify-models` (`uwbfall.tests.verify_bundle`) reports `BUNDLE CHECK: OK`, with "meta.json: OK (refit; differs only in non-predictive fields)".
- Predicting the fake test with this refit gives `754f6a3c…`, the reference CSV.
- Pass 1 got the same result at `b3716b8` (1087 s).

**Other documented targets**, run from the clone (patched Makefile)
- `make checks`: config invariants OK; drift-monitor sanity OK. 33 s.
- `make dry-run FAKE_SUB=…/audit_out/head/dryrun`:
  - `BUNDLE CHECK: OK`; 0 drift ALERTs; 119 rows; `SUBMISSION CHECK: OK`; CSV `754f6a3c…`; 61 s.
  - The integrator's evidence in `fake_test/submission/` was checked by sha256 before and after: **untouched** (F3).
- `make predict-test-fallback TEST_DIR=<fake> SUB=…/head/fb`: `BUNDLE CHECK: OK` on the fallback, `SUBMISSION CHECK: OK`, byte-identical to the integrator's `submission_fallback.csv`.
- `make srcv` was not re-run (75–110 min). The integrator showed its fold and val CSVs byte-identical to I1. The two from-scratch refits above cover the same fit and predict code.

**Negative tests of `verify_bundle`**
- One byte appended to `tcn_1.pt`: FAILED, exit 2.
- `meta.postproc` changed to 0.5: FAILED, exit 2.
- `make verify-models` on the tampered bundle: exit 2, so `predict-test` aborts.

## 2. Determinism

The fake test is the 5 val streams copied as `test_00..04`. All runs below use the HEAD clone's code.

| run | cache | threads | wall | peak RSS | CSV sha256 |
|---|---|---|---|---|---|
| cold (page cache dropped, new feature cache), repo `models/final` | cold | runtime 2, env 2 | **158.7 s** | **1.557 GB** | `754f6a3c…` |
| warm, repo `models/final` | warm | 2 | 52.3 s | 1.072 GB | `754f6a3c…` |
| warm, `/home/user/data_raw/uwbfall_models/final` | warm | 2 | 52.5 s | 1.100 GB | `754f6a3c…` |
| warm, env `OMP_NUM_THREADS=1` (config 2) | warm | 2 | 53.0 s | 1.079 GB | `754f6a3c…` |
| bundle copy with `runtime.threads=4` | warm | 4 | 60.5 s | 1.100 GB | `754f6a3c…` |
| bundle copy with `runtime.threads=1` | warm | 1 | 55.1 s | 1.073 GB | `f53f5150…` |
| installed venv entry point, no PYTHONPATH | warm | 2 | – | – | `754f6a3c…` |
| from-scratch `train-final` refit bundle | warm | 2 | – | – | `754f6a3c…` |
| `make dry-run` (patched) | team cache | 2 | 61 s total | – | `754f6a3c…` |
| integrator's dry-run, and pass-1 cold run | – | 2 | – | – | `754f6a3c…` |

- **The 1-thread variant is reproducible.** It gives the same `f53f5150…` as pass 1's cold 1-thread run. It has the same 119 rows and identical `stream_id`, `t_start`, `t_end` and `t_alarm`. 2 scores differ by exactly 0.001 (TCN float summation order, after rounding). Pass 1 showed that this leaves the official score unchanged.
- **The knob is the bundle's `runtime.threads`**, which predict enforces with `torch.set_num_threads` and `threadpoolctl`. The environment variable is irrelevant. SUBMISSION.md now says so (F7).
- **Features are thread- and cache-invariant.** The new cold cache is byte-identical to pass 1's cold cache (`audit_cache_p2`). Against the team cache, every array is identical; only the stored `sid` label differs (`test_03` vs `val_03`, same content hash).
- **No cross-stream dependence.** test_02 predicted alone in its own folder gives exactly its 31 rows from the 5-stream batch. So does the same file named `stream_00.h5` (id test_00, compared after relabelling).

## 3. `validate_predictions` on the dry-run CSV

`reports/round2/audit/val_csv.py` calls the official `abc2027.evaluate.validate_predictions` with the folder's official `stream_durations`. I ran it on `…/head/pred_cold.csv`, which is identical to `fake_test/submission/submission.csv` and to the patched `make dry-run` output. Result: **OK**.
- 119 records; header exactly `stream_id,t_start,t_end,score,t_alarm`; no CR characters;
- 5/5 streams (34/31/31/18/5 rows); 0 duplicate rows;
- scores 0.574–0.996;
- all 119 `t_alarm` values in [t_start, t_end];
- durations 1.00–4.40 s (median 1.88 s).

`uwbfall.tests.check_submission`, inside `make dry-run`, also printed `official validate_predictions OK` and `SUBMISSION CHECK: OK`.

## 4. Rule compliance

| check | how (pass 2) | result |
|---|---|---|
| Package unmodified | `e1lib.package_manifest()` before any run and after all runs, including both refits | `fb0e24027626640206472c8663f6bc1ab6aa1010e430fce42396ac7957b41aaf` both times, equal to `work/E1_pseudo_test/manifest_before.txt`. `find -newermt '2026-10-13 21:20'`: no file. **PASS** |
| Limitation of that manifest | code read | It content-hashes every code and CSV file, but for `.h5` and `.npy` only name, size and mtime (about 10 GB). An in-place edit that kept size and mtime would be invisible. Consistent with that, the val CSV reproduces bit-exactly from the h5 files. |
| Package `__pycache__` | `find` | `src/abc2027/__pycache__` and `scripts/__pycache__` are still present, from round 1. They are generated bytecode and no source changed, but strictly they are writes into the package. F8 (owner action). |
| No external or LIARA data, network or pretrained weights | `grep -rniE 'liara\|https?://\|urllib\|requests\.\|wget\|curl \|torch\.hub\|torchvision\|huggingface\|pretrained\|download'` over `uwbfall srcv analysis work Makefile`, including REVIEW's new code | **0 hits**. All models are trained from scratch: HGB, a TCN with fixed seeds, LightGBM segment classifier, PCA. **PASS** |
| Libraries open source | imports of `uwbfall` | numpy, scipy, pandas, scikit-learn, torch, h5py, lightgbm, threadpoolctl, plus the package's own `abc2027`. **PASS** |
| Transductive cross-stream ADL mask OFF | `final.json`, `reference.json`, both shipped `config.json`, `config.py` | `adl_duplicate_mask: false` everywhere. `config.py` pins it in the forced-value table and `check_configs` passes. **PASS** |
| No cross-stream information at predict time | AST read of `load_bundle`, `stream_probs`, `_boundaries`, `predict` (no label, events-CSV or composition access; the only "events" are the predicted ones), plus the empirical single-stream test in §2 | **PASS**. Drift monitors pool over test streams, but they only print and write JSON. |
| No test labels | none exist; `streams_events.csv` is read only by `train` and `score`; `composition_of` only by fit, retune and the drift reference | **PASS** |
| Val labels | The deployed bundle trains on train+val. Val is labelled package data, so this is allowed. Its operating point `[0.4, 3, 0.8, 0.6]` comes from its own 4-fold training OOF; the refit reproduced 0.9389. The round-2 choices were made on train OOF or SR-CV (R2_EVAL, I1, FINAL). | **PASS.** Process note, as in pass 1: a few frozen candidates were scored on val in round 1, and C3 → FINAL was accepted despite val 0.9646 → 0.9524. That is val exposure, not tuning. |
| `segments/` and the generator | SR-CV regenerates streams in memory | Package data, allowed. The deployed bundle uses only the 18 published h5 streams plus `streams_composition.csv` labels. |

**Disclosed interpretation risk (not a FAIL).** Several features exploit the published generator's structure:
- junction statistics at the 0.2 s crossfades;
- the sample-and-hold repeat share;
- the segment classifier gating on P(fall clip).

The package documents that construction, and the rules forbid only external (LIARA) recordings. SUBMISSION.md §6 now asks for this to be disclosed in the method description. The physics-only fallback exists in case the official test writer differs.

## 5. Predict RAM and time (5 streams, 2.05 h of radar, 4-CPU shared VM, 2 threads)

| condition | wall | peak RSS |
|---|---|---|
| cold: page cache dropped, empty feature cache (pass 2) | **158.7 s** | **1.557 GB** |
| cold (pass 1, `make features` running in parallel) | 143.2 s | 1.54 GB |
| cold (integrator) | 260 s | 1.60 GB |
| warm feature cache (pass 2, 6 runs) | 52.3–60.5 s | 1.07–1.10 GB |
| `train-final` for reference (pass 2) | 1312 s | 1.58 GB |

Everything is far below the 5 GB per-process budget. The cold time depends on machine load (I/O-bound extraction). SUBMISSION.md now quotes 2.5–4.5 min cold.

## 6. Fixes: status

| fix | problem (found in pass 1) | status (pass 2) |
|---|---|---|
| **F1** | `REPO ?= /home/user/challenge1` was hard-coded, so `make` from a clone silently ran the original repo's code | **Fixed.** Root `Makefile`: `REPO ?= $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))`, and the recursive `$(MAKE)` calls now pass `-f $(REPO)/Makefile REPO=$(REPO)`. `uwbfall/Makefile`: `REPO ?= $(abspath …/..)`. Verified with `make -n` and real runs from `/tmp` (`-f <clone>/Makefile`): every command uses the clone. In commit `0f2949b`. |
| **F2** | `verify-models` failed on a legitimate refit, because `meta.json` timings and git rev are in the manifest, so `predict-test` aborted after `make train-final` | **Fixed.** New `uwbfall/tests/verify_bundle.py`, used by `verify-models` and `verify-fallback`. All files must match exactly; `meta.json` may differ only in `timings`, `peak_rss_gb`, `git_rev` and `env`, and `postproc` must match. Verified on the shipped bundles, the pass-2 refit, the fallback, and two tampered copies, which fail with exit 2. In commit `0f2949b`. |
| **F3** | `dry-run` hard-coded `fake_test` paths and overwrote the integrator's evidence | **Fixed.** It now uses `$(FAKE)/streams/test` and `SUB=$(FAKE_SUB)` (default `$(FAKE)/submission`). Verified that the evidence was untouched. In commit `0f2949b`. |
| **F4** | `uwbfall/pyproject.toml` lacked `lightgbm` and `threadpoolctl`; no version lock for the pickled sklearn and LightGBM models | **Fixed, uncommitted.** Both dependencies added. New `requirements-lock.txt` with exact versions; pip dry-run shows that nothing would be installed. |
| **F5** | The root `README.md` was only `# challenge1` | **Fixed, uncommitted.** A short index: SUBMISSION, uwbfall README and lock, FINAL, AUDIT, REPORT and PLAN, plus key facts. |
| **F6** | `models/fallback_physics/` and runbook files were untracked | **Resolved by commit `31d30ed`.** |
| **F7** | SUBMISSION.md said 260 s cold; it presented `OMP_NUM_THREADS` as the determinism knob; the generator-structure disclosure was missing | **Fixed, uncommitted.** Cold time is now 143–260 s (2.5–4.5 min). `runtime.threads` is documented as the knob, with the audit's 1/2/4-thread results. The refit path now documents that `verify-models` accepts a refit. §6 now carries the generator-structure disclosure and the single-stream invariance. |
| **F8** (optional) | Round-1 bytecode in `<package>/src/abc2027/__pycache__` and `<package>/scripts/__pycache__` | **Open, for the owner.** Agents treat the package as read-only, so it was not touched. A human may `rm -rf` both folders; the manifest excludes `__pycache__`, so it stays `fb0e2402…`. The Makefile already sets `PYTHONDONTWRITEBYTECODE=1`. |

**For the lead to commit:**
- `README.md`
- `SUBMISSION.md`
- `uwbfall/pyproject.toml`
- `requirements-lock.txt`
- `reports/round2/AUDIT.md`
- `reports/round2/audit/head/` (40 KB of JSON and a script)

No prediction, model or config changed.

## 7. Remaining limitations (honest list)

- `make srcv` (SR-CV, 75–110 min) was not re-run by the audit. The SR-CV numbers in FINAL.md rest on the integrator's run. Their fit and predict code paths are covered by the two bit-exact from-scratch refits.
- The 18-stream cold `make features` was run only in pass 1 (496 s; 18/18 files byte-identical to the team cache). Pass 2 re-extracted only the 5 fake-test streams cold.
- Bit-exactness needs the locked versions (torch 2.14.1+cpu and others) and `runtime.threads = 2`. Any other thread count changes scores in the 3rd decimal at most, as observed for 1 thread.
- The package manifest does not content-hash the h5 and npy data (§4).

## Audit artefacts (outside git)

- `/home/user/data_raw/audit_clone_head`: pass-2 clone, about 69 MB.
- `/home/user/data_raw/audit_venv_head`: pass-2 venv.
- `/home/user/data_raw/audit_models_head/final`: pass-2 from-scratch train-final, about 6 MB.
- `/home/user/data_raw/audit_out/head/`: pass-2 CSVs, logs and official JSON.
- Pass 1:
  - `/home/user/data_raw/audit_clone`;
  - `/home/user/data_raw/audit_venv`;
  - `/home/user/data_raw/audit_cache` (457 MB, duplicate of the team cache);
  - `/home/user/data_raw/audit_cache_p2` (114 MB);
  - `/home/user/data_raw/audit_models`;
  - `/home/user/data_raw/audit_out`.
- All of these can be deleted after the commit. Run `rm -rf /home/user/data_raw/audit_{cache,cache_p2,venv,venv_head,clone}` to reclaim about 0.7 GB. Keep `audit_out` if the logs are wanted.

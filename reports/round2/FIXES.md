# Round-2 fixes (fixer pass)

**Bottom line.** I found and fixed one real blocker that no earlier role had found. On release day, a test stream
shorter than 2 s crashed `predict`, which meant no CSV for any of the five streams. Two smaller release-day
problems in the checker are also fixed. All three fixes are output-neutral on every stream of 5 s or more:

- the val CSV is byte-identical before and after (sha256 `d9f3ec22…`), and the official CLI still gives
  **0.9524 (64/2/3 @0.3)**;
- the dry-run CSV is byte-identical (`754f6a3c…`);
- the SR-CV numbers cannot change, because every SR-CV stream is at least 320 s long and training code is untouched.

The reviewer found no leakage and no metric error, so nothing in the model, the config or the bundles changed.
**Nothing is promoted.** Nothing is committed (the lead commits).

All numbers below come from commands run in this session. Logs are in `work/FIXES/logs/`.

## 1. Triage of the inputs

| source | issue | status after this pass |
|---|---|---|
| Auditor F1 (REPO hard-coded), F2 (verify-models fails after a refit), F3 (dry-run overwrote evidence) | blocked reproducibility | Already fixed in WIP commit `0f2949b`. **Re-verified here** (§3). |
| Auditor F4 (pyproject deps and lock file), F5 (root README), F7 (SUBMISSION wording) | packaging and docs | Already fixed in the working tree, uncommitted. **Re-verified here**: a copy of the working tree installs and runs (§3). |
| Auditor F8 (round-1 `__pycache__` inside the package) | cosmetic; the package is read-only for agents | Left to a human, as before. |
| Reviewer: leakage or metric error | none found | Nothing to fix. |
| Reviewer: replay gate 0.3 does not replicate on fresh compositions | not an error; neutral inside FINAL | No change (it would be new modelling). |
| Reviewer R1 (ADL subject prone to false alarms; transductive mask) | risk, needs organiser clearance | No change; the mask stays OFF. |
| Reviewer R2 (no per-segment repeat-spacing monitor) | not a blocker | Not done. The existing monitors for repeat share, rep_frac PSI and junction rate already fire on a writer without repeats or junctions (`make checks`). |
| **New, X1:** a stream under 2 s crashes `predict` | **blocker** (no submission) | **Fixed** (§2). |
| **New, X2:** `check-test` hard-fails on a short stream with no rows | release-day false failure | **Fixed** (§2). |
| **New, X3:** `check-test` crashes when the test folder is not named `test` | release-day false failure | **Fixed** (§2). |
| Docs: train-final time, disk space, short streams | stale or missing | SUBMISSION.md updated (§4). |

## 2. Fixes

### X1. Short remainder stream crashes predict (blocker)

**Why it can happen.** The official generator packs segments into 30-min streams, and the last stream of a split holds
what remains (`abc2027.streams._pack`). Val_04 is 5.2 min and train_12 is 18.9 min. The shortest segment in the
package is an ADL piece of 0.52 s, so in principle a remainder stream can be a fraction of a second long. The chance is
small: roughly 2 s out of a 30-min range, about 0.1%. The cost would be total, because a crash in one stream aborts
the whole run and no CSV is written.

**Evidence before.** I cropped val_00 into single-stream folders named `test` (`work/FIXES/make_short.py`) and ran
`predict` on them (`work/FIXES/run_short.sh`, `logs/short_sweep_before.log`):

| stream length | 0.52 / 0.8 / 1.0 / 1.2 / 1.5 / 1.8 s | 2.1 – 8 s | 9.17 s and longer |
|---|---|---|---|
| before | **crash** `ValueError: All arrays must be of the same length` (`features.stream_blocks`: the 2 s rolling windows) | 0 rows; check-test **FAILED** (X2) | rows; OK |
| after | 0 rows, warning logged; check OK | 0 rows; check OK (WARN) | **unchanged** |

**Fix.**

- `uwbfall/pipeline.py`: new `MIN_PREDICT_S = 5.0`. `predict` gives a stream shorter than this no rows and logs
  `WARNING only X s long`; the stream is not featurised.
  - Why this is safe: a stream is made of whole segments and the shortest fall clip is 7.32 s, so a stream under 5 s
    cannot hold a fall, and any row would be a false alarm.
  - The debug `cands` concat now tolerates an empty list.
- `uwbfall/drift.py`: `report` skips the reference checks when no stream was monitored. Before this, a batch made only
  of short streams crashed in `np.concatenate`.

**Regression test.** New `uwbfall/tests/check_short.py`, now part of `make checks`. It uses crops of 0.52, 1, 1.9,
4.9, 6 and 12 s, with the cache off, and must pass the official `validate_predictions`. Crops under 5 s must have no
rows, and the 12 s crop must find its fall.

| code | FINAL bundle | fallback bundle |
|---|---|---|
| HEAD `0f2949b` | **FAILED**: 0.52, 1.0 and 1.9 s raise `ValueError` (`logs/check_short_head.log`) | n/a |
| after | OK (`logs/checks_after.log`) | OK (`logs/check_short_fallback.log`) |

**Mixed release-shaped batch.** I also ran `make predict-test` on four 29-min streams (val_00..03 as test_00..03) plus a
1 s `test_04` (`logs/mixed_final.log`), and the same batch through `make predict-test-fallback`
(`logs/mixed_fallback_physics.log`). Results:

- Both runs end with `SUBMISSION CHECK: OK` and 0 drift ALERTs.
- The FINAL rows for test_00..03 are byte-identical to the dry-run rows for those streams.
- test_04 gets 0 rows (WARN).

### X2. `check-test` hard-failed on a short stream with no rows

Before, `uwbfall/tests/check_submission.py` required at least one row on every stream. Every short fixture from 2.1 to
8 s therefore ended in `SUBMISSION CHECK: FAILED streams without any prediction` (`logs/short_sweep_before.log`). Under
the runbook, that could have pushed the operator to the fallback for no reason.

Now a stream with no rows is a hard failure only when it is **10 min or longer** (at ~58 falls/h, zero detections there
is implausible). A shorter stream with no rows prints `WARN: no rows (allowed: stream shorter than 10 min)`. The other
hard checks are unchanged: the official validator, the header, and the expected ids.

### X3. `check-test` crashed when the test folder is not named `test`

`predict` forces the `test_XX` ids (`--sources dir:$TEST_DIR:test`), so its CSV is correct whatever the folder is
called. The checker, however, takes durations from the official `stream_durations`, which names streams of an
arbitrary folder `stream_XX`. Every row then failed as `unknown stream_id 'test_00'`, and the checker raised a
`ValidationError` traceback.

Now, when the folder is not train, val or test and expected ids are given, the checker maps `stream_XX` to the
expected prefix and prints a NOTE. On the same CSV with a folder named `release_x`, the HEAD checker gives a traceback
and the new one gives `SUBMISSION CHECK: OK`. SUBMISSION.md still recommends naming the folder `test`.

## 3. Verification after the fixes

| check | before (this session) | after |
|---|---|---|
| `make verify-models verify-fallback` | BUNDLE CHECK OK ×2 | unchanged (no bundle touched) |
| `make checks` | configs OK, drift monitor check OK | configs OK, drift OK, **SHORT-STREAM CHECK OK** (38 s) |
| `make reproduce-val`: val CSV sha256 | `d9f3ec22…`, byte-identical to I1 G7N4 | `d9f3ec22…`, byte-identical |
| official CLI on val (`--ignore-flagged`) | 0.9524, 64/2/3 | **0.952381**; F1 @0.1/0.3/0.5 = 0.977 / 0.962 / 0.917; 64/2/3 @0.3; median start error 0.055 s; 0.976 FA/h (`work/FIXES/val_after_official.json`) |
| `make dry-run` (val as test_00..04): CSV sha256 | `754f6a3c…`, 119 rows, 0 ALERT, CHECK OK | `754f6a3c…`, identical |
| working-tree copy (tracked plus uncommitted files, exported with `git ls-files -co`), `make -f <copy>/Makefile verify-models dry-run` | – | every command uses the copy's PYTHONPATH; BUNDLE OK ×2; CSV `754f6a3c…` (`logs/clone_dryrun.log`) |
| `pip install` of the copy's `uwbfall` in a fresh venv (system site packages, no build isolation), `pip check`, installed `uwbfall predict` on the mixed batch | – | "No broken requirements found"; CSV byte-identical to `make predict-test` (`5cc47ec1…`) |
| SR-CV pooled numbers (FINAL 0.9221, C3 0.8861) | – | unchanged by construction: the shortest SR-CV test stream is 320 s, far above 5 s, and `fit` and `retune` are untouched. Not re-run. |
| determinism | – | no numeric path changed. Every CSV above is byte-identical to its earlier hash. |

Every step is warm-cache: predict takes 40–60 s for the 5-stream sets. Peak RSS is about 1.15 GB.

## 4. Documentation changes

- `SUBMISSION.md` §1: keep the folder named `test` (X3 is a safety net only). A disk note: about 5.8–5.9 GB free;
  test streams ~2 GB plus any archive; what can be deleted. The last stream can be short.
- `SUBMISSION.md` §2 and §4: short-stream behaviour (X1, X2), and that a stream under 5 s with no rows is not a reason
  to switch to the fallback. `make train-final` takes about 22 min (the audit measured 1312 s), not 15–18 min.
- `Makefile`: `make checks` now also runs `check_short`.
- `uwbfall/README.md`: the checker and the new test.
- No release-day command changed.

## 5. Files changed or added (uncommitted)

- Modified:
  - `uwbfall/pipeline.py` (X1)
  - `uwbfall/drift.py` (X1)
  - `uwbfall/tests/check_submission.py` (X2, X3)
  - `Makefile` (checks)
  - `SUBMISSION.md`
  - `uwbfall/README.md`
- Added:
  - `uwbfall/tests/check_short.py`
  - `work/FIXES/` (`make_short.py`, `run_short.sh`, `README.md`, `logs/`, `val_after_official.json`)
  - this file
- Still uncommitted from the audit: `README.md`, `uwbfall/pyproject.toml`, `requirements-lock.txt`,
  `reports/round2/AUDIT.md`, `reports/round2/audit/head/`.
- Scratch outside git (about 50 MB, deletable): `/home/user/data_raw/fixes_{sub,short,short2,mixed,cache}`.

## 6. Limitations

- I tested only a few failure modes of the release-day input: short streams, the folder name and a mixed batch. I did
  not test a test writer that differs in dtype, chunking or dataset names. Predict reads only the `cir` dataset, as the
  official evaluator does.
- The 5 s cut-off is a structural argument (no fall clip is shorter than 7.32 s). It is not fitted to data. Streams of
  5 s or more behave exactly as before.
- The reviewer's risks R1–R3 still apply: expected test score about 0.91, 80% interval [0.85, 0.96]. This pass did not
  change the model.

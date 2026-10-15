# Release-day submission runbook (ABC 2027 UWB fall challenge)

**System:** `uwbfall/configs/final.json`, frozen by the round-2 integrator. It equals the round-2 reference
(`DEFAULT_CONFIG`; `tests/check_configs` asserts this) and contains every change that passed the promotion rule:

1. 0.75 TCN (3 members) + 0.25 HGB blend (round 1).
2. Annotator replay with agreement gate **0.3** (R2_EVAL).
3. Segment-classifier gate p·(0.7 + 0.3·P(fall clip)) and a 4 s time NMS (I1).

I2 and I3 promoted nothing. Evidence is in `reports/round2/FINAL.md`.

**Model:** `/home/user/data_raw/uwbfall_models/final`. It is trained on all 18 labelled streams (train + val). The
operating point comes from the bundle's own 4-fold training OOF; val was not used for any choice. A runnable copy with
sha256 is in `models/final/` (3.7 MB; `MANIFEST.sha256`, `manifest.json`).

> **Rule compliance (organisers' Q2 answer):** submit with `VARIANT=final_norep` unless the organisers confirm that
> frame-level sample-and-hold statistics are allowed; use `VARIANT=final_strict` if they forbid the replay dedup.
> Every `make` command below accepts `VARIANT=...`. See `reports/COMPLIANCE.md`.

## 0. Before release day (already done; re-run only if something changed)

```bash
cd /home/user/challenge1
make checks          # config invariants + drift-monitor sanity (synthetic drift must alarm, clean data must not)
make dry-run         # rehearsal: val streams copied as test_00..test_04 under /home/user/data_raw/fake_test
```

The dry run passed: `reports/round2/FINAL.md` §4, logs in `work/FINAL/logs/dryrun_*_r2.log`.

## 1. Get the test streams

Download the test release. Place the five files as `stream_00.h5 … stream_04.h5` in **one folder named `test`**. The
official ids are taken from the folder name (`test_XX`). Either location works:

* `<package>/streams/test/` (default `TEST_DIR`). Copy them in yourself; the agents never write into the package.
* any other folder, e.g. `/home/user/data_raw/test_release/streams/test`, passed as `TEST_DIR=...`.

Keep the folder name `test`: the official evaluator derives the ids from it. (Predict forces the `test_XX` ids
anyway, and since FIXES.md X3 `check-test` also accepts another folder name, with a NOTE.)

Disk: about 5.9 GB was free on 2026-10-15. The five streams take about 2 GB, plus the archive if you download one, plus
about 25 MB of feature cache per stream. If space is short, delete the dry-run fixture `/home/user/data_raw/fake_test`
(1.7 GB; `make fake-test` rebuilds it) and the audit and fixer scratch folders listed in AUDIT.md and FIXES.md.

Check the files: `ls -la $TEST_DIR` should show 5 files of about 400 MB. The last stream of a split holds the
remaining segments and can be much shorter (val_04 is 5.2 min). Then run
`python3 -c "import h5py,sys; [print(f, h5py.File(f)['cir'].shape) for f in sys.argv[1:]]" $TEST_DIR/stream_*.h5`.
Each shape must be `(3, T, 368)`.

## 2. One command

```bash
cd /home/user/challenge1
make predict-test TEST_DIR=/path/to/streams/test        # SUB defaults to /home/user/data_raw/submission
```

It does the following:

1. `verify-models` (`uwbfall.tests.verify_bundle`) checks the sha256 of every FINAL bundle file against
   `models/final/MANIFEST.sha256`. `meta.json` may differ only in timings, peak RSS and git rev (a legitimate refit), and its
   operating point `postproc` must match.
2. `python -m uwbfall predict --models $FINAL --sources dir:$TEST_DIR:test --out $SUB/submission.csv`:
   * extracts features (cold cache: about 25–50 s per 30-min stream; 2.5–4.5 min in total cold for 5 streams);
   * runs HGB + TCN, the clip gate, decode, replay and NMS;
   * runs the **official `validate_predictions`** before the CSV is written;
   * **prints the drift monitors** and writes `$SUB/submission_drift.json`.
3. `check-test` runs `uwbfall.tests.check_submission`. It checks the official validator, the exact header
   `stream_id,t_start,t_end,score,t_alarm`, and ids test_00..test_04 with at least one row for every stream of
   10 min or more. A shorter remainder stream with no rows is only a WARN (FIXES.md X2). It also prints
   per-stream events/h, durations and score ranges.

If `/home/user/data_raw/uwbfall_models/final` is missing or fails `verify-models`, use the repo copy, which carries the same
sha256: `make predict-test TEST_DIR=... FINAL=/home/user/challenge1/models/final`. If neither is available, run
`make train-final` (about 22 min; the audit measured 1312 s), then `make predict-test` as above. It is bit-reproducible: a from-scratch refit
matched 9/10 files exactly, `meta.json` differs only in timings and git rev, and `verify-models` accepts it
(AUDIT.md §1).

When the test streams are inside the package, the equivalent literal command is
`python -m uwbfall predict --models /home/user/data_raw/uwbfall_models/final --split test --out <csv>`. In the dry run
it gave a byte-identical CSV (`reports/round2/FINAL.md` §4).

**Time / memory (dry run, 5 streams = 2.05 h, 2 threads):**
- Cold feature cache and cold page cache: **143–260 s** wall depending on machine load (integrator 260 s; audit 143 s and
  159 s), 1.56–1.60 GB peak RSS. Feature extraction is most of it.
- Warm cache: 52–69 s for predict, about 60 s for the whole `make dry-run` including the checks; 1.1 GB peak RSS.

## 3. Checks before uploading

| check | expected | where |
|---|---|---|
| `SUBMISSION CHECK: OK` | hard requirement | end of `make predict-test` |
| 5 streams, roughly 35–70 events/h each (dry run 37–69; train truth ~58 falls/h) | WARN outside [10, 150]; a short remainder stream can show any rate | check_submission output |
| `DRIFT MONITORS … 0 ALERT(s)` | 0–1 ALERT is fine | predict output, `$SUB/submission_drift.json` |
| `$SUB/predict.log` contains no Traceback | | |

Upload `$SUB/submission.csv`.

### Drift monitors (label-free, computed within the test streams; `uwbfall/drift.py`)

| monitor | rule (ALERT) | dry run (val as test) |
|---|---|---|
| predicted events/h | outside [0.5, 1.6] × the bundle's training-OOF rate (55.0/h) | 58.1 |
| detected junctions/h | outside [0.6, 1.6] × training (250/h) | 243 |
| share of predictions starting ≤ 5 s after a junction | < training share − 0.20 (0.84) | 0.78 |
| repeated-frame (sample-and-hold) share | outside [0.5, 1.5] × training (0.236) | 0.235 |
| per-radar median raw-energy level (log10) | outside training per-stream range ± 1 decade | inside |
| per-radar PSI of block log motion energy, log frame-difference energy, repeated-frame fraction (10 training-quantile bins) | > 0.25 (0.10–0.25 WATCH) | ≤ 0.009 |

`tests/check_drift.py` shows that the monitors react to synthetic drift:
- CIR scaled ×10: level and PSI alarms.
- Repeats removed: repeat-share and rep_frac alarms.
- Junction-free writer: junction-rate alarm.
- Clean val data: none.

Raw-energy *shape* is deliberately not monitored. It depends on the room: val vs train PSI reached 1.2 with no real
drift.

## 4. Fallback

**Trigger.** The predict output says `RECOMMEND physics-only fallback` (2 or more ALERTs), or the check shows a
streams-level anomaly. Examples: a stream of 10 min or more with 0 predictions, or events/h far outside the dry-run range.
A stream under 5 s gets no rows by design. It is shorter than any fall clip (7.32 s minimum), and the features need
at least 2 s. Before FIXES.md X1 such a stream crashed the whole predict. It is not a reason to switch.

**What the fallback is.** `uwbfall/configs/fallback_physics.json`:
- prototype "physics" columns only, with no EMA-junction, t_since/t_until, segment or PCA-fingerprint features;
- no junction-based clip gate;
- replay with junction-free `loc` crops only;
- same blend, bands, gate 0.3 and 4 s NMS.

It needs no generator artefact, but it is **weaker**:
- Trained on the 13 train streams, it scores official val **0.9239** (59/1/8 @0.3), against FINAL's 0.9524 (64/2/3).
- Its train+val training OOF is 0.920, against 0.939 for FINAL.

See `reports/round2/FINAL.md` §5. It is a switch for a broken test writer, not an alternative to try.

```bash
make train-fallback          # ~15 min; ONLY if the prebuilt bundle is missing: /home/user/data_raw/uwbfall_models/fallback_physics
                             # (prebuilt; sha256 copy in models/fallback_physics/, check: make verify-fallback)
make predict-test-fallback TEST_DIR=...   # -> $SUB/submission_fallback.csv (+ checks)
```

**Decision rule:**
* **0–1 ALERT:** submit the FINAL CSV.
* **2 or more ALERTs:** look at which monitors fired.
  * Repeat share or junction rate fired: the generator's structure changed, which is what the clip gate and structure
    features depend on. Submit the fallback.
  * Only PSI or level fired, while events/h and the near-junction share are normal: submit FINAL, because the
    detections are still coupled to the clip structure.
* If one submission slot remains and both CSVs pass the checks, prefer FINAL unless the structural monitors fired.

## 5. Reproduction targets (`Makefile`)

| target | what | time |
|---|---|---|
| `make features` | warm the feature cache for train, val and (if present) test | ~13 min cold |
| `make train-final` | FINAL bundle from scratch (train+val, `final.json`), bit-reproducible | ~22 min |
| `make predict-test` | release-day command (above) | 1 min warm, 2.5–4.5 min cold |
| `make reproduce-val` | `final.json` on the 13 train streams, val CSV, official score; must be byte-identical to the I1 G7N4 bag-0 val CSV (0.9524) | 1 min with the bundle present (`uwbfall_models/final_train13`), else +20 min |
| `make srcv` | clean SR-CV run of `final.json` (4 folds + official val), then `srcv-table` (final table + promotion rule vs C3) | ~75–110 min on one lane (fold fits take 11–23 min each) |
| `make checks`, `make dry-run` (runs `make fake-test` first, a 1.6 GB copy of val) | invariants and rehearsal | 1–2 min (+ copy) |

Environment: `PYTHONPATH=<repo>:<package>/src`, `OMP_NUM_THREADS=2`. The Makefile sets both and finds `<repo>` from its own
location, so `make -f <clone>/Makefile ...` runs that clone's code. Exact library versions: `requirements-lock.txt`
(torch 2.14.1+cpu, scikit-learn 1.9.1, lightgbm 4.7.0, ...).

Determinism is governed by the bundle's `config.json` `runtime.threads` (2), which predict enforces with
`torch.set_num_threads` and `threadpoolctl`. The `OMP_NUM_THREADS` env var does not change the output. In the audit,
`runtime.threads` 2 and 4 gave byte-identical CSVs. With 1 thread the rows and times were identical, and 2 of 119 scores
moved by 0.001 (same official score). Do not edit `runtime.threads`.

## 6. Compliance

- Only package data and code are used: streams, `segments/`, the generator for SR-CV, and `abc2027.annotate.propose_fall`
  for boundaries.
- No external data and no pretrained weights. All libraries are open source.
- Every feature is computed within one stream. The transductive cross-stream ADL duplicate mask is **OFF**
  (`transductive.adl_duplicate_mask = false`). Predicting one stream alone gives the same rows as in the 5-stream batch
  (AUDIT.md §2).
- Disclose in the method description: several features use the published stream generator's structure (0.2 s crossfade
  junction statistics, sample-and-hold repeat share, a segment classifier gating on P(fall clip)). The package documents
  that construction. The physics-only fallback does not depend on it.

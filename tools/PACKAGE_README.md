# UWB-FallTrack (ABC 2027): code and model for verification

Variant: **`{{VARIANT}}`**. The submitted predictions were produced with the model in `models/{{VARIANT}}/` and the
commands below. Raw 3-radar UWB streams go in; one CSV row per detected fall comes out
(`stream_id,t_start,t_end,score,t_alarm`).

## Contents

| path | what |
|---|---|
| `uwbfall/` | the whole pipeline: feature extraction, training, prediction, boundaries, drift monitors, CLI (`python -m uwbfall`) |
| `uwbfall/configs/{{VARIANT}}.json` | frozen configuration of the submitted system |
| `models/{{VARIANT}}/` | trained model (train + val, 18 streams), `MANIFEST.sha256` |
| `models/fallback_physics/` | physics-only fallback, used only if the drift monitors flag a changed stream structure |
| `Makefile` | one target per step (`predict-test`, `train-final`, `verify-models`, `reproduce-val`, `checks`, `dry-run`) |
| `run_all.sh` | one command for the whole local run (environment, dataset, features, training, checks, dry run) |
| `requirements-lock.txt` | exact library versions (Python 3.13) |
| `results/` | the submitted CSV, its drift report and log (present when the package was built after the test run) |
| `SHA256SUMS`, `VERSION` | file hashes, source revision |

Code comments mention development folders (`work/`, `analysis/`) where each part was first prototyped. They are not
part of this package and nothing imports them.

## Environment

* Linux (or WSL2), Python 3.13, `make`, `bash`.
* `uv venv -p 3.13 .venv && source .venv/bin/activate`
* `uv pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu`
* The challenge package `abc2027_uwb_train_v1` (its `src/` provides the official evaluator and
  `abc2027.annotate.propose_fall`). Default location: `/home/user/data_raw/abc2027_uwb_train_v1`. Elsewhere, pass
  `DATA=<folder that contains abc2027_uwb_train_v1>` to every `make` command (feature cache, models and outputs then go
  there too), or `PKG=<package folder>`.
* CPU only is enough. Results are bit-reproducible with these versions; the bundle fixes 2 compute threads
  (`runtime.threads`).

## 1. Predict the test streams (the submitted CSV)

```bash
make predict-test VARIANT={{VARIANT}} TEST_DIR=/path/to/streams/test      # folder named "test" with stream_00..04.h5
# -> /home/user/data_raw/submission/submission.csv   (change with SUB=...)
```

This checks the model against its sha256 manifest, predicts every stream on its own, runs the official
`validate_predictions`, prints the drift monitors and checks the CSV (ids `test_00..test_04`, header). About 3–5 min on
4 CPU cores. Without `make`:

```bash
PYTHONPATH=.:$PKG/src ABC_ROOT=$PKG python3 -m uwbfall predict --models models/{{VARIANT}} \
    --sources dir:/path/to/streams/test:test --out submission.csv
```

## 2. Reproduce the model from scratch

```bash
make train-final   VARIANT={{VARIANT}}     # ~20 min on 4 cores; trains on the 13 train + 5 val streams
make verify-models VARIANT={{VARIANT}}     # compares the retrained files with models/{{VARIANT}}/MANIFEST.sha256
```

Every model file is bit-identical to the shipped one on CPU with the locked versions. `meta.json` may differ only in
timing, environment and provenance fields that prediction never reads; the operating point must match, and
`verify-models` checks that. (`bash run_all.sh cpu --variant {{VARIANT}}` runs everything, including the dataset
download.)

## 3. Validation

```bash
make reproduce-val VARIANT={{VARIANT}}     # trains on the 13 train streams, predicts val, official scorer
```

Official evaluator, ranking score = mean event F1 at tIoU 0.1/0.3/0.5 with `--ignore-flagged`:

| | val (trained on train only) | subject-disjoint CV (4 folds, 253 falls) |
|---|---|---|
| official baseline | 0.315 | – |
| **`{{VARIANT}}`** | {{VAL}} | {{SRCV}} |

The subject-disjoint CV regenerates streams from the package's `segments/` with the official generator, so that every
fold's test subjects are unseen. It was used only for model selection; the shipped model is trained on the published
train and val streams only.

## Method (short)

1. **Features**, per 0.2 s block, from each stream on its own: 64 physics/structure features (motion and frame-difference
   energy per radar, static-profile changes, crossfade-junction context, segment statistics) + 48 range-band energies.
2. **Detectors:** gradient-boosted trees (scikit-learn HGB) and a 3-seed dilated temporal CNN. Blend 0.75 TCN + 0.25 HGB.
3. **Segment gate:** a classifier of the segment between two junctions scales each score:
   p × (0.7 + 0.3 · P(fall segment)).
4. **Decoding:** threshold/merge on the blended curve, operating point chosen on the training folds' out-of-fold
   predictions only; 4 s non-maximum suppression.
5. **Boundaries:** the package's own annotator (`abc2027.annotate.propose_fall`) is re-run on six crops around each
   detection; median start/end, agreement gate 0.3, 1.0 s minimum duration.

## Data and rules

* Only the challenge package is used (streams, `segments/`, generator, annotator). No other data, no pretrained
  weights; all libraries are open source.
* Each test stream is processed independently. Nothing is matched or compared across streams or with the train/val
  segments, and the transductive cross-stream option in the code is off (`transductive.adl_duplicate_mask = false`).
* `{{VARIANT}}` does not use frame-repeat (sample-and-hold) statistics as features.
  - `final_norep`: before the annotator replay, a frame identical to the previous frame of the same radar is dropped,
    to recover the radar's true frame sequence (`boundary.dedup`).
  - `final_strict`: drops nothing.

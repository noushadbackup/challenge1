# E1: pseudo-test bank

This package regenerates many new "val-like" streams from the validation segments in `<package>/segments`. It uses the official planning code with new seeds and test settings. Only features and CSVs are stored (no h5). A scorer runs the official evaluator on them.

Every number below comes from scripts in this directory, run in this session. The logs are `logs_*.txt` and the outputs are in `results/`.

## What was built

| file | role |
|---|---|
| `e1lib.py` | `val_groups()` is a read-only copy of `build_streams._groups()` for val. `build_stream()` replicates `build_streams._write_stream` line by line in memory: 0.2 s crossfade, event protection, same rounding. `extract()` replicates `analysis/extract_features.process` one radar at a time, with the same keys and dtypes. `package_manifest()` hashes the package. |
| `check_parity.py` | Rebuilds the 5 **real** val streams from `streams_composition.csv` and compares them with the published h5. It also compares the extractor output with the team cache `feats/val_XX.npz`. |
| `gen_bank.py` | For each seed: `prepare_split("test", val falls/ADL/others, rng=default_rng(seed))` then `finish_plan(bundle, 57.1864)`. Then `build_stream`, `extract`, and `/home/user/data_raw/pseudo/feats/p<seed>_<XX>.npz` plus `events_<seed>.csv` and `composition_<seed>.csv`. Split `"test"` gives `adl_uses=2` and `ignore=true` for low-quality / needs_review falls, as in the official test. The density 57.1864 is the `configs/streams.yaml` target; after crossfades it yields 58.0 falls/h, as published. |
| `scorer.py` | Official `abc2027.evaluate` (`validate_predictions`, `load_ground_truth(ignore_flagged=True)`, `evaluate`) with durations = `e_raw.shape[1]/50` taken from the npz. `--verify` compares it with the official CLI. |
| `score_bank.py` | Scores one prediction CSV covering `p<seed>_<XX>` streams, split by split. Gives mean ± sd, plus a paired delta against a reference CSV (`--ref`). **This is the entry point for other executors.** |
| `eval_prototype.py` | Uses the `analysis/prototype_detector.py` recipe unchanged: same features, same HGB params with random_state=0, 4-fold GroupKFold-by-stream OOF on the 13 train streams, and the same 80-point post-processing grid chosen on train OOF. It reports the official val score and then scores every pseudo split. |
| `bank_stats.py` | Distribution parity of the bank against the real val streams. |

## Verification (all passed)

1. **Generator replica is bit-exact** (`logs_parity_builder.txt`). All 5 real val streams rebuilt from the composition CSV match the published h5 bit for bit on all 3 radars. Event times and composition times are identical too: 34+31+31+18+5 = 119 events.
   - One CSV caveat: `source_t0/t1` are printed with 3 decimals. A time on an exact half frame (for example 0.350 s = 17.5 frames) is then ambiguous under `np.round`, so the check tries ±1e-6 s nudges and keeps the one whose interior frames match.
   - The bank itself uses full-precision floats, so this ambiguity does not arise there.
2. **Extractor replica is bit-exact** (`logs_parity_extract.txt`, val_04; val_00 was also checked interactively). All 7 keys (`e_raw, e_mot, e_diff, mot_map, amp_1s, prof_1s, prof_02`) are bit-equal to the team cache. Peak RSS is about 1.45 GB per worker, against about 3.5 GB for the original extractor.
3. **Scorer equals the official CLI** (`logs_scorer_verify.txt`). Npz durations equal the h5 durations. The full JSON report is identical to the CLI's for `sample_submission.csv` (0.315091) and for the prototype val CSVs: full 0.883951, physics 0.790323, structure 0.875912.
4. **Prototype reproduction**: `eval_prototype.py` gives train OOF 0.9009 and val **0.8840** (official CLI) for full, and 0.8471 / 0.7903 for physics. This is identical to `reports/prototype/summary.json`.
5. **Package untouched**: the package manifest sha256 is the same before and after (`fb0e2402…`; `manifest_before.txt`). No file in the package is newer than the start of the run. All runs used `PYTHONDONTWRITEBYTECODE=1`.

## Bank contents

There are 10 seeds (1001–1010). Each seed is one complete pseudo val split: 5 streams (4 of about 29.4 min plus 1 short one), 2.05 h, 119 falls, 67 evaluable. In total that is **50 streams, 20.5 h, 1.6 GB** in `/home/user/data_raw/pseudo/`. Generation took about 1 min per seed with 2 workers.

Parity with the real val streams (`logs_bank_stats.txt`):

| | real val | bank (10 splits, range) |
|---|---|---|
| falls/h | 58.08 | 57.99–58.06 |
| evaluable falls | 67 | 67 |
| junctions/h | 284.6 | 279.9–288.2 |
| ADL time share | 0.717 | 0.716–0.722 |
| median pre / post margin (s) | 3.20 / 7.54 | 3.10–3.19 / 7.24–7.70 |
| median ADL piece (s) | 11.46 | 11.42–12.13 |

KS tests of the pooled bank against real val: pre-margin p=0.66, post-margin p=0.87, ADL piece length p=0.45.

## Results: prototype on the bank

The model is trained on the 13 train streams, with post-processing tuned on train OOF only. Every number uses the official evaluator code with `--ignore-flagged` semantics. The unit is one pseudo split, comparable to the 5-stream val set; n = 10 splits.

| feature set | val ranking F1 (official CLI) | **bank ranking F1, mean ± sd (n=10)** | bank min–max | bank F1 @0.1/0.3/0.5/0.7 | bank median start err @0.3 | bank FA/h @0.3 |
|---|---|---|---|---|---|---|
| physics | 0.7903 | **0.7849 ± 0.0156** | 0.759–0.812 | 0.833/0.802/0.720/0.521 | 0.235 s | 4.10 |
| full | 0.8840 | **0.8766 ± 0.0197** | 0.851–0.908 | 0.926/0.890/0.813/0.541 | 0.225 s | 3.27 |

- **Paired full − physics**: +0.0917 ± 0.0310 on the bank, positive in 10/10 splits, against +0.0936 on val.
- **Val against the bank**: val lies +0.37 sd (full) and +0.35 sd (physics) above the bank mean, so the official val streams are a typical draw.
- **Per stream** (n=50; files `results/per_stream.csv` and `results/per_seed.csv`):
  - full: 0.882 ± 0.063 on the 40 long streams; 0.75 ± 0.31 on the 10 short streams of about 5 falls each.
  - Per-stream sd is dominated by the short streams. Use the per-split numbers for decisions.

## How to run

```bash
cd /home/user/challenge1/work/E1_pseudo_test
export PYTHONDONTWRITEBYTECODE=1
python3 check_parity.py 0 1 2 3 4            # replica checks on the real val streams (~2 min, 1.7 GB)
python3 gen_bank.py 1001 1002 ...            # ~1 min / seed, 2 workers x ~1.45 GB, ~160 MB / seed
python3 bank_stats.py                        # parity table
python3 scorer.py --verify                   # scorer == official CLI
python3 eval_prototype.py physics full       # ~2 min, 0.85 GB
python3 score_bank.py my_preds.csv [--ref ref_preds.csv] [--out r.json]
```

To score your own model on the bank:

1. Load `/home/user/data_raw/pseudo/feats/p<seed>_<XX>.npz`. The keys and dtypes are the same as in `/home/user/data_raw/feats`. For the prototype features, `eval_prototype.build_unlabelled` shows how.
2. Write one CSV with `stream_id = p<seed>_<XX>`.
3. Run `score_bank.py`.

Labels for training-style use, which should not normally be needed, are in `events_<seed>.csv` and `composition_<seed>.csv`, in the same format as the package CSVs plus a `seed` column.

## Honest limitations

- **It measures context noise, not subject noise.** Every split contains the **same 119 val falls** from the same 4 val fall subjects and the same 30 val ADL files. Only the margins, ADL cuts, ordering and crossfade neighbours change. The sd of about 0.02 is therefore a lower bound on test variance: X4 estimates about 0.05 from subject draws. It is a tool for checking robustness to stream construction and for paired comparisons, not an estimate of the test score. X3/X4's subject-disjoint regenerated CV (SR-CV) is the complement; `e1lib.build_stream` and `extract` can be reused for it as they are.
- **Val is not independent of the bank.** It shares all fall and ADL content with the bank. Do not tune on the bank and then report val as a holdout.
- **The official val plan cannot be reproduced exactly.** Only the generation *logic* is replicated. The public seed does not reproduce the ADL pieces or the stream order (X5).
- **Some generator differences are not modelled.** The official test is written by an unshipped `scripts/rebuild_test.py`. Any differences it has from `_write_stream` cannot be reproduced.
- **The density is an assumption.** It is the `configs/streams.yaml` target (57.1864). This matches the published 58.08 falls/h after crossfades, but the test density could differ.
- **The prototype is seed-sensitive.** It uses HGB with `early_stopping='auto'` (X5), so `random_state=0` is fixed as in `prototype_detector.py`. Model-seed variance (X5 measured sd 0.012 on val) is **not** included in the bank sd above.
- **No CIs from bootstraps here.** The ± values are the sd across the 10 splits.

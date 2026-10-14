# E3: complex I/Q and radar-physics features

**Bottom line.** On train leave-streams-out CV, adding the complex I/Q feature group (`full+iq`) improves the prototype `full` model consistently: +0.012 to +0.017 ranking F1, and the gain holds across 3 fold maps and 3 seeds. On the 5 official val streams the gain is **not confirmed**. Over 4 paired runs the delta is -0.010 / +0.007 / +0.015 / -0.005 (mean +0.002), which is inside the noise from threshold choice and seed. Recommendation: keep `iq` as a candidate feature view or ensemble member and judge it in the team's subject-disjoint CV. Do not promote it on this evidence. The tie-break cost is real: median start error is about 0.05-0.10 s worse.

Every number below comes from code in this directory, run in this session. Each event metric uses the official evaluator: the CLI for every val number, and the same `abc2027.evaluate` functions imported in-process for the train-OOF grid search.

## What was built

`extract_iq.py` streams the raw h5 one radar at a time. It uses float32, peaks at 0.54 GB RSS per process, runs in about 25 s per 30-min stream with 2 jobs, and writes 25 MB in total to `/home/user/data_raw/feats_iq/`. The physics choices follow the data verdict (F1, F5, repeat finding) and X1 P1:

- z = I + jQ, with I = values 0..183 and Q = values 184..367.
- Mat range zone only: R29171 (idx 0) bins 40-68 (D3), and R29444 / R30722 (idx 1/2) bins 5-21.
- Bit-exact held frames (about 25-27%) are dropped. All lag-1 statistics are computed between consecutive **distinct** frames, using the true gap.
- Clutter removal: a centred 25-distinct-frame (~0.6 s) moving mean on the distinct sequence. There is no 3 s EMA, and no phase unwrapping or integration.
- The extractor stores per-block sums (0.2 s grid, the same as `prof_02`), so the features can be re-aggregated at any scale.

Feature groups in `iq_experiment.py`. Each group is computed per radar plus the max over radars, at 0.2 s and 1 s scales:

| group | features |
|---|---|
| `iq` (86 cols) | sample-and-hold rate; raw lag-1 decorrelation `log(1-coh)`; high-passed energy; phase concentration (1 - circular variance) of the moving part; share of high-pass energy with \|dphi\| > pi/2 (fast or aliased motion); rate-normalised frame-difference energy over all 368 values and over the zone only (the jump of a hold is divided by its length); signed mean Doppler (rad/s) and unit-phasor coherence; high-pass range centroid, spread and centroid velocity (x 5.14 cm/bin); ±1/±2 s context of three key signals |
| `zone` (16 cols) | past-vs-future static change at 1 s / 2 s / 5 s, restricted to each radar's zone bins and normalised by the zone's median norm (from cached `prof_02`) |
| `seg` (7 cols) | segment-relative signals over the prototype's jump-to-jump segment: value minus segment max, time to the segment argmax, mean hold rate in the segment |

## Protocol

The protocol is identical to `analysis/prototype_detector.py`: the same blocks, the same HGB hyper-parameters, and the same 60-point post-processing grid tuned on train OOF only.

- **CV**: leave-streams-out over the 13 train streams, with 3 fold maps (the prototype's GroupKFold(4) plus 2 shuffled stream-to-fold maps). The post-processing is chosen on the mean over the 3 OOFs.
- **Main comparison**: deterministic, with `early_stopping=False`. X5 showed that the prototype's `early_stopping='auto'` makes it seed-dependent.
- **Seed check**: `early_stopping='auto'` with seeds 0/1/2 and 1 fold map, which is the prototype's exact setting.
- **Pipeline check**: `full_es_s0` reproduces the prototype exactly (train OOF 0.9009, val 0.8840 = 0.948/0.919/0.785/0.504).
- **Val use**: val was used once per reported arm, after the arm was chosen on train.

## Results

### Deterministic (early stopping off), train CV with 3 fold maps

| arm | n feat | train OOF rank F1 (mean of 3 maps) | per map | val rank F1 | val F1 @.1/.3/.5/.7 | val TP/FP/FN @.3 | val med \|start err\| | val FA/h |
|---|---|---|---|---|---|---|---|---|
| full (reference) | 64 | 0.8890 | .902/.891/.875 | **0.8922** | .941/.912/.824/.515 | 62/7/5 | 0.23 | 3.42 |
| **full+iq** (train-selected) | 150 | **0.9059** | .913/.896/.908 | 0.8825 | .921/.892/.835/.504 | 62/10/5 | 0.32 | 4.88 |
| full+zone | 80 | 0.9025 | .910/.896/.902 | not run | | | | |
| full+iq+zone | 166 | 0.9051 | .913/.891/.911 | 0.8725 | .926/.897/.794/.544 | 61/8/6 | 0.28 | 3.90 |
| full+iq+zone+seg | 173 | 0.9045 | .897/.908/.909 | not run | | | | |
| physics | 25 | 0.8478 | .841/.852/.850 | 0.8142 | .852/.820/.770/.459 | 50/5/17 | 0.19 | 2.44 |
| physics+iq+zone | 127 | 0.8911 | .888/.893/.893 | 0.8136 | .866/.835/.740/.551 | 53/7/14 | 0.26 | 3.42 |

Paired stream-cluster bootstrap on the map-0 OOF (13 streams, B = 5000; `compare.py`):

| arm vs reference | delta | 90% CI | P(delta > 0) |
|---|---|---|---|
| full+iq vs full | +0.0117 | [+0.0003, +0.0242] | 0.95 |
| full+zone vs full | +0.0083 | [-0.0064, +0.0248] | 0.78 |
| full+iq+zone vs full | +0.0114 | [-0.0012, +0.0238] | 0.93 |
| full+iq+zone+seg vs full | -0.0051 | [-0.0233, +0.0133] | 0.33 |
| physics+iq+zone vs physics | +0.0472 | [+0.0148, +0.0739] | 0.99 |

### Prototype setting (early stopping 'auto'), 3 seeds

| arm | train OOF (s0/s1/s2, mean) | val rank F1 (s0/s1/s2, mean) | val F1@0.5 mean | val med start err |
|---|---|---|---|---|
| full | .9009/.8919/.8931, **0.8953** | .8840/.8838/.8806, **0.8828** | 0.788 | 0.22/0.27/0.27 |
| full+iq | .9002/.9126/.9169, **0.9099** | .8905/.8990/.8759, **0.8885** | 0.803 | 0.32/0.26/0.32 |

The paired val delta per seed is +0.0065 / +0.0152 / -0.0047, a mean of +0.006.

### Diagnostics (val is not used for any choice)

**Threshold plateau** (`thr_curve.py`, smooth 5, min_dur 0.8, gap 0.2; train OOF / val):

| threshold | full | full+iq |
|---|---|---|
| 0.3 | .889 / .905 | .913 / .882 |
| 0.4 | .902 / .892 | .903 / .894 |
| 0.5 | .878 / .869 | .900 / .910 |

`full+iq` has a flatter train plateau. On val, the threshold alone moves either arm by ±0.02, which is larger than the arm difference.

**Val error breakdown at θ0.3** (`fp_breakdown.py`):
- `full+iq` (train-chosen threshold 0.3): 10 FPs, of which 6 are inside Chute04 fall recordings (post-fall or pre-fall motion fragments) and 3 are in Autre segments.
- `full` (threshold 0.4): 7 FPs.
- The FNs are the same 5 Chute01/Chute04 events in both, except one swap.

**Per-feature AUC on train** (`results/feature_auc.csv`; evaluable fall blocks vs fall-recording margins / ADL / Autre-Empty):
- The radar-to-mat mapping is confirmed. On unflagged D3 (P3_3) falls, the R29171 far-zone features reach `iq_hpE1_r0` 0.991/0.988/0.978, `iq_mz1_r0` 0.994/0.988/0.954 and `zst10_r0` 0.959/0.971/0.996.
- The same features on R29444 and R30722 are about 0.3-0.5 for D3.
- For D2, the R30722 features (`iq_hpE1_r2`, `iq_mz1_r2`) reach 0.96-0.997. For D1, `iq_md1_r1` (rate-normalised diff on R29444) reaches 0.98/0.97/0.95.
- Signed mean Doppler is uninformative (AUC 0.43-0.62), as expected from aliasing at about 0.35-0.46 m/s unambiguous speed.
- On the far zone of R29171, the high-|dphi| share is *lower* in D3 falls (AUC 0.10-0.34 against all negatives), so it works as an inverted cue.

## Honest limitations

- **Train CV is leaky.** Subjects and ADL files are shared across train streams (data verdict). The +0.012-0.017 train gain may partly be subject memorisation by the bigger feature set. The val result (4 subjects, 67 evaluable falls, CI about ±0.06) can neither confirm nor reject a gain of this size. This needs the subject-disjoint SR-CV that X3/X4 propose.
- **Tie-break cost.** Median start error rises from 0.22-0.27 s to 0.26-0.32 s with `iq`. This is irrelevant if boundaries come from the annotator replay (X1/X2), which is the likely final design.
- **D3 not measured.** No held-out-subject D3 measurement was made. Val has no evaluable D3, and stream-grouped CV does not hold out P3_3. The zone features are the right D3 hedge on per-feature AUC (0.99 for P3_3), but whether they lift held-out-P3_3 recall (X3: 0/16 for the prototype) is **unverified**.
- **Segment features hurt.** The `seg` group is built on the prototype's crude p97 jump segments, and it reduced train CV (-0.005). It should be re-tried on the variance-normalised junction detector.
- **Partial X1 coverage.** No STFT or Lomb-Scargle spectra were tried. Multi-lag coherence and the |dphi| > pi/2 share stand in for Doppler spread.
- **Excluded arms.** `full+zone` and `full+iq+zone+seg` were not scored on val, because they were not train-best and val is only for final candidates.
- **Bug found and fixed in this session.** The prototype's `cols_for` picks up any column whose name starts with `seg_`. My first run therefore leaked the segment features into every arm, including the baseline. That run is archived in `results/buggy_seg_leak/` and is not used; the features were renamed to `sg_` and all arms were re-run. Anyone extending the prototype should avoid the prefixes `static`, `diff`, `jump`, `ema`, `t_since`, `t_until`, `seg_` and `fp` for new columns.

## How to run

The script runs from this directory. Peak RSS is about 0.9 GB, and `OMP_NUM_THREADS=2` keeps it within the 2-core limit.

```bash
cd /home/user/challenge1/work/E3_iq_features
python3 extract_iq.py 2                                   # ~6 min, writes /home/user/data_raw/feats_iq/*.npz
OMP_NUM_THREADS=2 python3 iq_experiment.py full full+iq full+zone full+iq+zone full+iq+zone+seg physics physics+iq+zone --reps 3   # train CV only (~20 min)
OMP_NUM_THREADS=2 python3 iq_experiment.py full full+iq full+iq+zone physics physics+iq+zone --reps 3 --val          # + one official val score each
OMP_NUM_THREADS=2 python3 iq_experiment.py full full+iq --val --es --seeds 0 1 2            # prototype setting (seed 0 == 0.8840)
python3 compare.py full_r3 full+iq_r3 full+zone_r3 full+iq+zone_r3 full+iq+zone+seg_r3      # paired stream bootstrap
python3 collect.py; python3 feature_auc.py; python3 thr_curve.py full_r3 full+iq_r3; python3 fp_breakdown.py preds/val_full_r3.csv preds/val_full+iq_r3.csv
```

Outputs:
- `results/<tag>.json`: config, train OOF including per-stream official tp/fp/fn, and val metrics.
- `results/summary.md`: the collected results table.
- `preds/`: OOF and val probabilities, val CSVs, and the official val JSONs.

## Suggested next steps for the integrator

1. Run `full` vs `full+iq` (and `+zone`) inside the subject-disjoint SR-CV with at least 5 seeds, under the X4/X5 promotion rule. If they pass, use `iq` as a second feature view or ensemble member (an equal-weight probability mean) rather than as a replacement.
2. Evaluate the `zone` group on a held-out-P3_3 fold. This is the case it is designed for.
3. Feed the zone-restricted static change and the rate-normalised distinct-frame motion (`Md`, `Mz`) to the annotator-replay selector (X1 P0). They are the closest analogues of `annotate.py`'s state and motion signals.

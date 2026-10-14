# V4 — Leakage & shortcut auditor

**Role:** I tried to break the lead's findings F3, F7 and F8 with my own code, using different methods from the lead's scripts. I also audited the validation protocol for leakage and looked for shortcuts the generator leaves in the data.
**Scratch code:** `/home/user/data_raw/agents/V4/` (`f8_leak.py`, `f3_junction.py`, `f7_fp.py`, `dup.py`, `oracle.py`, `bias.py`). Every number below comes from these scripts, run in this session.

## What I ran

| script | method | data |
|---|---|---|
| `f8_leak.py` | pandas crosstabs over `streams_events.csv` and `streams_composition.csv`; overlap audit of source files and subjects; pre-margin per fall | CSVs |
| `oracle.py` | **official evaluator** (`--ignore-flagged`) on oracle submissions: all GT events; GT plus 17 s junk on flagged falls; Gaussian boundary jitter; fixed-duration and shifted variants | val |
| `f3_junction.py` | **official** `abc2027.background.subtract_background(tau=3)` and `baseline.frame_energy` on the raw val h5 files. The lead used his own lfilter. FAs come from the evaluator's own `_match_stream` on `sample_submission.csv` (= energy baseline) | 5 raw val streams |
| `f7_fp.py` | log amplitude of the 1 s static profile → StandardScaler + PCA(30) + logistic regression, and a separate nearest-centroid score with no training. Train→val. Blocks within ±0.6 s of junctions are dropped. Margins = fall segment minus fall ±1.5 s. Also scored at segment level and with range-bin subsets | `amp_1s` cache |
| `dup.py` | share of frames that repeat the previous one (`e_diff==0`) and of zero frames, per segment | `e_raw`, `e_diff` cache |
| `bias.py` | signed start and end errors of the prototype's matched predictions | `reports/prototype/*.csv` |

## Votes

| F | vote | conf | by code | evidence |
|---|---|---|---|---|
| F1 I/Q split | agree | 0.75 | no | I did not test it. The argument is plausible, and the paper's figure of 184 bins at 5.14 cm agrees. |
| F2 timing | agree | 0.85 | partly | Recomputed pre-margin (fall start − segment start): 24 falls have pre-margin < 0.05 s, so the overall min of 0 is right. **Evaluable** falls: min **1.0 s**, p5 1.34 s, median 3.2 s. Other numbers not rechecked. |
| F3 EMA junction spikes | agree (magnitude differs) | 0.85 | **yes** | Official 3 s EMA on raw val, 583 junctions. Peak in [j−0.2 s, j+1 s] / median ADL energy (≥5 s from a junction): median **63.5×** (p25 22.8, p75 94.1). adl→adl 31×; any junction that switches corpus (adl↔fall, fall→fall) is **103–130×**, which is as high as D1/D2 fall peaks (117× / 122×). 85% of junctions exceed 10×. Decay below 3× reference: median **5.1 s**, p90 10.6 s. Energy baseline at θ=0.3: 100 FAs, **90.0% start within 1 s of a junction** (exactly the lead's figure), 92% within 2 s. Median start is +0.14 s after the junction. FAs fall in fall-rec 50, adl 42, other 8. "84×" depends on which reference you pick. The qualitative claim holds. |
| F4 junction detector | agree | 0.7 | no | Not re-run. |
| F5 dataset → radar | agree | 0.75 | partly | In my F7 run, the single-radar fingerprint AUC is highest on radar 2 (0.958) > radar 1 (0.913) > radar 0 (0.872). That is consistent with the claim but does not test it directly. Composition shows Autre = D1/D2 positions and Empty = D3. |
| F6 static change | agree | 0.7 | no | Not re-run. It fits the README ("bounds placed from change in static CIR profile"). |
| F7 corpus fingerprint | **partially** | 0.85 | **yes** | My method is stronger than the lead's: fall-corpus vs ADL val block AUC **0.956** (segment level 0.952, nearest centroid 0.955). Margins only (fall ±1.5 s excluded): **0.960**. Far bins only (60–183, more than 3 m from the mat): **0.929**, so this is a room/session background, not the person. **However**, fall-rec vs Autre/Empty gives **0.69–0.70** with logistic regression and 0.64 with the centroid score, not 0.56 (≈chance). There are only 6 val Autre/Empty segments, so this number is very noisy. Treat "≈chance" as unproven. |
| F8 D3 flagged / 67 evaluable | **agree** | 0.97 | **yes** | Crosstab: train D1 31/120, D2 38/119, D3 104/120 flagged (86.7%). Val D1 7/40, D2 6/40, D3 39/39 (100%). Evaluable val = 33 + 34 + 0 = 67. The official evaluator on an oracle of all 119 val GT events gives TP 67, FP 0, 52 ignored predictions. The `ignore` column is false for all 478 events, so `--ignore-flagged` is required. `streams.py` sets `ignore` for test only when the event is low quality or needs review, which matches the claim. Note: 13 val D1/D2 falls are also ignore zones, not just D3. |
| F9 no activity context | agree | 0.9 | code read | `streams.finish_plan` does `rng.shuffle(falls+others+adl)`, so neighbours are random by construction. |
| F10 prototype 0.884, no leakage | **partially** | 0.75 | **yes** | Re-scoring `val_full.csv` with the official evaluator gives **0.8840** [.948/.919/.785/.504]. Val really is clean: **0** fall subjects shared train/val, **0** ADL files shared. **But** the train OOF used to tune post-processing is leaky (see N1). With only 67 events, ±0.03–0.04 is noise. |

## New findings

### N1 — The train OOF CV is not independent: ADL content repeats across streams and fall subjects span most streams
* `ADL_USES = {"train": 3, "val": 2, "test": 2}`: every ADL file is cut 3 times in train with different offsets. **All 90/90** train ADL files appear in more than one train stream. **80.9%** of train ADL seconds (77.9% in val) are an **identical copy** of source frames that also appear in another stream of the same split (same `.npy` slice).
* Each of the 18 train fall subjects appears in **9–12 of the 13** train streams.
* So the `GroupKFold(4)` by stream in `prototype_detector.py` holds out neither the subject nor the ADL content. Thresholds and post-processing tuned on OOF are optimistically biased: the model has seen the same ADL frames. OOF 0.901 vs val 0.884 suggests the bias is modest so far. It will grow with higher-capacity models (CNN/GBM on raw profiles can memorise the duplicated ADL).
* **Action:** CV folds should group by fall subject index (`Test_P*_k`) **and** by ADL source file (`activities/<act>_<k>`). An easy way is to fold on subject number k for both corpora, or to regenerate training streams per fold from disjoint `segments/`. Keep val as the only untouched holdout for model selection.

### N2 — The ranking is almost entirely boundary precision; event duration barely matters (oracle tests, official evaluator, 67 val falls)
| oracle variant | rank F1 | F1 @.1/.3/.5/.7 |
|---|---|---|
| Gaussian jitter on start and end, σ=0.25 s | 0.995 | 1.0/1.0/.985/.842 |
| σ=0.5 s | 0.917 | .985/.946/.821/.49 |
| σ=0.75 s | 0.773 | .905/.809/.606/.227 |
| σ=1.0 s | 0.629 | .785/.66/.442/.161 |
| exact start, **fixed 1.74 s duration** | 0.985 | 1/1/.955/.716 |
| exact centre, fixed 3.0 s | 0.920 | 1/1/.761/.313 |
| constant +0.5 s shift | 0.920 | 1/1/.761/.119 |

* The prototype's 0.884 is about what σ≈0.55 s of boundary noise gives. Duration regression is worth ≤0.015, so the effort belongs in **start/centre localisation and systematic-bias calibration**. A constant 0.5 s offset alone costs 0.08.
* Prototype val matched errors: start median +0.01 s, end median +0.11 s, |start| median 0.23 s. In an exploratory run, widening every val event by 0.26 s on each side gave **0.909** (F1@.5 0.785→0.844, F1@.7 0.504→0.326). That shift came from a mismatched OOF grid point (`tmp_train.csv` is the *last* grid configuration, not the best one). So this is an **unvalidated hint**, not a result. Add an explicit "event width / start offset" parameter to the post-processing grid and tune it on properly grouped OOF.

### N3 — Generator shortcuts that are safe to exploit, and one risk
* **Evaluable falls never start at a junction.** 24 falls have pre-margin < 0.05 s. Their before-margin was clipped to 0, so `protect=0`: a hard splice with no crossfade lands exactly at fall onset. **All 24 are flagged.** For every evaluable fall, onset is **≥ 1.0 s** after the junction (p5 1.34 s, median 3.2 s). Prior for test: in a fall-like segment, put the onset 1–8 s after the junction and never at it.
* **The ignore zone is wide and free.** An oracle plus 17-s-wide junk predictions over each flagged fall gives TP 67, **FP 0** (all 52 junk predictions ignored). Any unmatched prediction that touches a flagged fall ±2 s costs nothing, so the detector can run a low threshold near D3-looking / low-quality fall segments.
* **The duplicate-frame rate is another corpus fingerprint.** About 24–27% of frames are nearest-neighbour repeats. The per-segment repeat rate separates fall recordings from **Autre/Empty with segment AUC 0.90 (train and val)**, and from ADL 0.63–0.66. It complements the static fingerprint exactly where F7 is weak.
* **Risk:** the official test streams are written by an unpublished `scripts/rebuild_test.py` with a hidden seed. Junction-timing and fingerprint shortcuts assume the same generator (margins 3–8 / 5–10 s, 0.2 s fade). The README's wording supports this, but it is not guaranteed. Keep a physics-only fallback (val 0.790) and make sure no single shortcut feature dominates. Check feature importance or run an ablation without the fingerprint (the lead's "structure" row: 0.876).

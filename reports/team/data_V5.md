# V5: Statistician (significance, confidence intervals, sample sizes). Verification report

Role: check whether the lead's numbers survive an independent method, put error bars on them, and say which differences are real.
All numbers below come from code I ran in this session. Scripts and outputs are in `/home/user/data_raw/agents/V5/`.

## What I ran

| script | what it does (different method from the lead where possible) | output |
|---|---|---|
| `f10_stats.py` | My own re-implementation of the official greedy matcher (it reproduces the official F1 exactly). Cluster bootstrap of ranking F1 over 5-minute windows (25 units) and over streams (5 units). Per-stream scores. Bias of matched start/end times. | `f10_stats_val.json`, `gt_outcomes_val.csv` |
| `paired_boot.py` | Paired cluster bootstrap (5-min windows, 4000 resamples) of the ranking-F1 difference between two prediction files on val. | stdout (quoted below) |
| `f10_seeds.py` | Re-runs the lead's full pipeline (same features, OOF tuning on train only) with 4 model seeds and with LightGBM. Also scores every post-processing grid point on val, to see how flat the surface is. Checks that train and val subjects do not overlap. | `f10_seeds.json`, `f10_seeds.log`, `val_*.csv` |
| `f9_perm.py` | F9 by a within-stream permutation test (5000 shuffles of the segment order, G statistic and total variation distance), not the lead's chi-squared test. Also tests chunk length and the kind of the previous segment. | stdout |
| `f4_junction.py` | My own junction detector. It compares 1 s windows on either side of each 0.2 s boundary, with a 0.2 s guard for the crossfade, using the complex 0.2 s profiles. I tried two normalisations: s1 = relative jump, s2 = jump divided by the variance inside the windows. Scored by boundary AUC, and as an event detector (NMS 3 s, match within ±1.5 s, threshold tuned on train only). | `f4.json` |
| `n_boundary.py` | Boundary-offset calibration. A constant start/end shift is tuned on train OOF events only, applied once to val and scored with the official evaluator. Also computes the IoU ceiling that the 0.2 s grid alone imposes. | `b_va.csv`, `s_va.csv`, `*.json` |

## Verification of F9, F10 and F4

### F9: activity before a fall is random. **Confirmed.**
- The segment before a fall is an ADL chunk 389 times, another fall clip 82 times, Autre/Empty 3 times, and the stream start 4 times.
- Within-stream permutation test for the ADL label before a fall: G = 10.95, **permutation p = 0.485**; TV distance 0.067, p = 0.436. By split: train p = 0.44, val p = 0.77. The lead's chi-squared test gives p = 0.494, so the two methods agree.
- The ADL chunk before a fall is no longer or shorter than other chunks: median 11.80 s vs 11.82 s, Mann-Whitney p = 0.94. The chunk after a fall gives p = 0.56.
- The kind of the current segment (adl / fall / other) does not predict that the next segment is a fall: chi-squared p = 0.10. There are 82 fall-to-fall joins against 94.9 expected under permutation (95 % interval 80 to 110), p = 0.105.
- Conclusion: there is no usable context prior from the activity sequence.

### F10: prototype scores 0.884 on val. **Confirmed. The noise statement is understated.**
- My own matcher reproduces the official result exactly: F1 0.948 / 0.919 / 0.785 / 0.504, ranking 0.8840.
- No train/val subject overlap (0 shared subjects). The PCA and the post-processing are fitted on train only.
- **Seed robustness** (full pipeline re-run, post-processing re-tuned on train OOF each time): HGB seeds 0/1/2/3 give **0.884 / 0.884 / 0.881 / 0.871** (mean 0.880, sd 0.006). **LightGBM with the same protocol gives 0.900.** The result is not a lucky seed.
- **Uncertainty.** The cluster bootstrap over 25 five-minute windows gives a **95 % CI of [0.824, 0.940]** (sd 0.029). F1@0.5 is [0.68, 0.88] and F1@0.7 is [0.38, 0.62]. The stream-level bootstrap gives [0.862, 0.920], but with only 5 streams it underestimates the spread. The lead's "±0.04 is within noise" is too optimistic: the honest half-width is about **±0.06**.
- **Paired comparisons** (paired bootstrap):
  - full vs structure: +0.008, 95 % CI [−0.051, +0.065], P(≤0) = 0.38. **The background-fingerprint gain is not significant.**
  - structure vs physics: +0.086 [+0.029, +0.147], P(≤0) < 0.001. **The segment-structure gain is real.**
  - full vs physics: +0.094 [+0.026, +0.167].
- **Post-processing sensitivity.** Over the lead's 60-point grid, the val score of seed 0 ranges up to 0.891 with a median of 0.847. The choice of post-processing moves the score about as much as the feature sets do.
- Per-stream ranking F1: val_00 0.894 (21 evaluable falls), val_01 0.847 (18), val_02 0.877 (19), val_03 0.909 (5), val_04 1.000 (4). Two of the five val streams carry almost no information.

### F4: junctions are detectable from static-profile jumps. **Confirmed, and a better normalisation exists.**
- s1 (relative jump, close to the lead's idea): boundary AUC 0.955 train / **0.934 val**. As an event detector it gives val P/R 0.864 / 0.688.
- **s2 (jump divided by within-window variance):** boundary AUC **0.985 train / 0.974 val** (0.979 against boundaries more than 2 s from a junction). As an event detector with the threshold tuned on train: train P/R/F1 0.958 / 0.865 / 0.909, **val 0.922 / 0.811 / 0.863**. This is better than the lead's 0.83 / 0.82.
- **Recall depends on the junction type.** The junctions that matter are detected well: on val, adl→fall 0.91 (99), fall→adl 0.98 (99) and fall→fall 0.94 (18). ADL→ADL junctions are the hard ones, at 0.72 (355), and they are mostly harmless because no fall is near them.

## Votes on all ten findings

| finding | vote | conf. | verified by my code | evidence / comment |
|---|---|---|---|---|
| F1 I/Q layout | agree | 0.85 | no | Three independent arguments in the report. V1 and V2 confirmed it at sample level. I did not re-test it. |
| F2 timing structure | agree | 0.8 | partly | My F9 script reproduces the composition counts (1844 ADL chunks, 478 fall clips, 24 others). I did not recompute the margins. |
| F3 EMA junction spike | agree | 0.75 | no | Plausible from the definition of a 3 s causal EMA. The 84× and 90 % numbers were not re-derived by me. |
| F4 junction detectability | **agree** | 0.9 | **yes** | Own detector: val AUC 0.974, val P/R 0.92 / 0.81 at ±1.5 s. Fall-related junctions have recall 0.91 to 0.98. |
| F5 mat → radar | agree | 0.7 | no | Consistent with V3's localisation work. I did not test it. |
| F6 fall vs motion peak; static change | partially | 0.6 | no | The static-change AUCs look right, and V3 reproduced them. The "motion peak ~1.5 s before onset" claim depends on the EMA, and V3 showed that frame-difference energy peaks inside the fall. The finding's last sentence says the same. |
| F7 corpus fingerprint | partially | 0.6 | partly | My paired bootstrap shows that adding the fingerprint to the prototype gives **+0.008, CI [−0.05, +0.07]: not significant**. V1 and V2 found that frame-timing separates fall clips from Autre/Empty, so "≈ chance" holds only for the amplitude cue. |
| F8 D3 flagged, 67 evaluable val falls | agree | 0.95 | **yes** | D3 flagged: train 104/120 (0.867), val 39/39. Evaluable val falls: 67 (D1 33, D2 34). Note that **D1/D2 are also flagged 26 % / 32 % in train and 18 % / 15 % in val**. |
| F9 activity before a fall is random | **agree** | 0.95 | **yes** | Permutation p = 0.485 (train 0.44, val 0.77). Chunk length p = 0.94. Previous-segment kind p = 0.10. |
| F10 prototype 0.884 | **agree** | 0.85 | **yes** | Reproduced exactly with my own matcher. 4 seeds: 0.871 to 0.884. LightGBM: 0.900. No subject leakage. But the 95 % CI is [0.82, 0.94], not ±0.04. |

## New findings

### N1: The val set cannot tell apart improvements smaller than about 0.05. Select models on pooled leave-stream-out CV.
- Evidence: with 67 evaluable val falls, the 95 % CI of the ranking F1 is ±0.058. Paired differences between two reasonable systems have a bootstrap sd of 0.030, so a gain must be about **+0.06** before it is significant at 5 % on val alone.
- Of the prototype's ablations, only "structure" passes this test. The fingerprint step (+0.008) does not.
- Changing only the model seed moves the val score by 0.013 (0.871 to 0.884). Changing only the post-processing choice moves it by 0.04 (grid median 0.847, best 0.891).
- Implication:
  - Do model selection and post-processing tuning on **leave-one-stream-out CV over train + val pooled** (18 streams, 253 evaluable falls). This shrinks the standard error by about √(253/67) ≈ 1.9×.
  - Report every comparison as a **paired bootstrap** difference, not as two point estimates.
  - Average seeds, or bag models, before trusting a 0.01 gain.
  - Never pick the post-processing on val.

### N2: The ranking metric rewards slightly wider intervals. A train-tuned padding gives +0.020 on val.
- Evidence (`n_boundary.py`): I tuned a constant shift on train OOF events only. The best was **start −0.3 s, end +0.2 s**, which widens each interval by 0.5 s. Applied once to val:
  - ranking F1 went **0.884 → 0.904**, mainly from F1@0.5 (0.785 → 0.830); F1@0.3 went 0.919 → 0.933.
  - Paired bootstrap: +0.020, CI [−0.005, +0.045], P(≤0) = 0.076. This is consistent in direction, and on train OOF it was 0.901 → 0.914.
- Cost: F1@0.7 drops from 0.504 to 0.311 (θ = 0.7 is not in the ranking), and the median start error rises from 0.22 s to 0.40 s (this is only a tie-break).
- Why it works: the ranking averages θ = 0.1/0.3/0.5. Once a prediction contains the true fall, IoU = |GT| / |pred|, so a modest over-coverage still clears 0.5. Under-coverage or a misaligned interval loses IoU on both sides.
- The 0.2 s block grid by itself is not the limit: snapping the true events to the grid still gives IoU ≥ 0.82 (median 0.945).
- The real limit is boundary scatter. On val, the end-error IQR is [−0.10, +0.46] s and the start-error IQR is [−0.14, +0.37] s, against [−0.18, +0.15] s and [−0.14, +0.16] s on train OOF. The correlation between predicted and true duration is only 0.31 on val.
- Implication:
  - Make the boundary padding an explicit hyperparameter, tuned in CV for the **mean F1 over 0.1/0.3/0.5**, not for IoU or θ = 0.7.
  - A regression head for start/end offsets should be trained with a loss that matches this objective.
  - Watch the tie-break: if team scores are close, re-centre the start.

### N3: Use the noise-normalised junction statistic. The relative jump wastes recall on junctions into and out of fall clips.
- Evidence:
  - Dividing the 1 s before/after complex profile jump by the variance inside the windows (s2) raises val boundary AUC from 0.934 to 0.974, and val event recall from 0.69 to 0.81 at higher precision (0.92).
  - For the junctions the model needs, recall on val becomes adl→fall 0.91 and fall→adl 0.98.
  - The prototype's `structure_features` uses a fixed 97th-percentile threshold on `jump_max`. That is the structure step, and it is the only significant gain (N1).
- Implication: replace `jump_max` with the s2 statistic, and give the model the threshold-free "time since/until the strongest s2 peak". This is a cheap change to the component that carries the +0.086.

## Files
`/home/user/data_raw/agents/V5/`: `f10_stats.py`, `paired_boot.py`, `f10_seeds.py` (+ `f10_seeds.json`, `f10_seeds.log`), `f9_perm.py`, `f4_junction.py` (+ `f4.json`), `n_boundary.py` (+ `b_va.csv` base, `s_va.csv` padded val predictions).

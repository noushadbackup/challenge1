# Data team: consolidated verdict (round 2, L1 + L2)

This is the verdict the field experts and executors should rely on. It merges L1 (justifier) and L2 (skeptic) after a round-2 debate. Every point where the two leads differed was re-checked with code.

Sources:
- Round-1 reports: `data_L1_round1.md`, `data_L2_round1.md`, `data_V1..V8.md`.
- Round-2 scripts: `/home/user/data_raw/agents/L1/r2.py` (F4 AUC definitions, repeat-spacing cue, in-fall repeat drop) and `r2_f5.py` (F5 peak-bin definitions). Both were run in this session on cached features for all 18 streams.

## 1. Vote tally and final status

Votes are the 8 analysts' round-1 votes. "Final" is the joint L1+L2 status.

| ID | Agree | Partially | Disagree | Final | Corrected statement (short) |
|---|---|---|---|---|---|
| F1 | 8 | 0 | 0 | **confirmed** | 368 values = 184 complex range bins; I = 0..183, Q = 184..367. |
| F2 | 8 | 0 | 0 | **confirmed** (footnotes) | Pre-margin 0-7.86 s (median 3.03); post-margin 3.26-10.0 s (median 7.56); fall segment 7.32-18.64 s; Autre/Empty 77-120 s; duration median 1.74 s; 58.08 falls/h. Minimum fall-to-fall gap is 5.9 s end-to-start (7.4 s is start-to-start). Evaluable falls start ≥1.00 s after the junction. |
| F3 | 8 | 0 | 0 | **corrected** | The EMA junction spike is ~55x the ADL median (range 55-65x depending on reference; p10-p90 ~12-115x), not 84x. It decays below 3x in a median of ~5 s (p90 ~9-11 s), not ~8 s. 90% of energy-baseline FAs start within 1 s of a junction (confirmed). |
| F4 | 7 | 1 | 0 | **corrected** | Per-junction AUC is 0.975-0.977 val for the plain jump and **0.984-0.989 val** for the variance-normalised jump. Event-level at ±1.5 s: plain P 0.86 / R 0.70-0.71; variance-normalised P 0.96-0.97 / R 0.78-0.87. The quoted R 0.82 / P 0.83 is not reproduced. |
| F5 | 5 | 3 | 0 | **corrected** | Each mat is one radar/range cell. D1 → R29444 (idx 1); D2 → R30722 (idx 2); **D3 → R29171 (idx 0)** at far bins ~42-62. "D3 weak on all radars" is wrong. The peak bin depends on the definition (table in section 2). |
| F6 | 5 | 3 | 0 | **corrected** | Static before/after change vs ADL: AUC ~0.98 (D1), ~0.99 (D2), ~0.60 for all D3, ~0.965 for unflagged D3 (n=16). The "motion peak ~1.5 s before onset" is an EMA/junction artefact and is rejected. Frame-difference energy peaks inside the fall for ~87-90% of evaluable D1/D2 falls, median +0.7 s / +1.06 s after onset. |
| F7 | 3 | 5 | 0 | **corrected** | Static amplitude fingerprint, fall-recording margins vs ADL: val AUC ~0.94-0.98 (stronger than reported). Fall recording vs Autre/Empty: ~0.69-0.74, not chance, but rests on only 6 val segments. Adding it to the prototype gives no significant gain (+0.008, CI [-0.051, +0.065]). |
| F8 | 7 | 1 | 0 | **confirmed** (nuance) | 67 evaluable val falls (D1 33, D2 34, D3 0); train D3 104/120 flagged. Flagging is deterministic (`annotate.py` rules) and **subject-dependent**: Test_P3_3 is 20% flagged, the other 7 D3 subjects 100%. D1/D2 are also 15-32% flagged. `--ignore-flagged` is mandatory. |
| F9 | 8 | 0 | 0 | **confirmed** | Neighbouring activity is random (one `rng.shuffle` in `finish_plan`; permutation p≈0.40-0.49). |
| F10 | 6 | 2 | 0 | **confirmed** (caveats) | Val ranking F1 0.8840 (0.948/0.919/0.785/0.504; TP/FP/FN 62/6/5 @0.3; median start error 0.22 s). No val leakage. 95% CI ≈ [0.82, 0.94]. Train OOF used for tuning is leaky. Planning baseline ≈ **0.86** (rotation mean 0.861). |

## 2. Round-2 debate: points where L1 and L2 differed

### D-a. F4: does variance normalisation raise the AUC? (L2 right; L1 concedes)
- In round 1, L1 reported varnorm AUC 0.954 < plain 0.969, while L2 reported 0.989 > 0.977. The two leads used different positive definitions:
  - L1 counted every 0.2 s block within 0.3 s of a junction as a positive.
  - L2 took the per-junction maximum within ±0.4 s.
- `r2.py` computes both on the same statistic. Each cell gives the AUC for per-junction max / all blocks within 0.3 s:

  | Split | Plain jump | Variance-normalised |
  |---|---|---|
  | Train | 0.990 / 0.986 | **0.993** / 0.954 |
  | Val | 0.977 / 0.971 | **0.986** / 0.945 |

  Results are the same with or without a 0.2 s guard block (val varnorm 0.984 with the guard).
- The normalised statistic peaks more sharply, so its off-peak blocks score lower. Detection uses the peak, so the per-junction definition is the relevant one, and **the variance-normalised statistic is better on AUC as well as on event precision.**
- Event recall at the train-chosen threshold varies with the threshold grid and peak spacing: L1 0.775, L2 0.799, V2 0.873, V5 0.811. Precision is consistently 0.92-0.97.
- Recall on junctions that bound fall clips is 0.89-1.00. Most misses are ADL→ADL junctions.

### D-b. F5: D1 peak bin, 13 (L1) or ~9 (L2)? (both right; the bin depends on the definition)
`r2_f5.py`, events on their dominant radar:

| Dataset (radar) | argmax of raw mean motion | argmax of ADL-normalised mean | argmax of ADL-normalised max |
|---|---|---|---|
| D1 (R29444) | 9 (IQR 5-14) | 13 (9-21) | 13 (9-20) |
| D2 (R30722) | 11 (8-20) | 11 (9-14) | 11 (9-14) |
| D3 (R29171) | 61 (46-61) | 46 (42-62) | 59 (43-62) |

- Raw motion energy favours the nearest bins.
- Per-bin ADL normalisation shifts D1 to about bin 13.
- D3 is bimodal by subject: ~44-47 for P3_1, P3_2, P3_5 and P3_9; ~61-62 for P3_3, P3_6 and P3_8.
- **Practical rule:** feature windows should cover bins ~5-21 on R29444/R30722 and ~40-64 on R29171.

### D-c. Frame-repeat cue vs Autre/Empty: L1's 0.78-0.81 or L2's ~1.0? (L2 right at segment level; important nuance on time scale)
`r2.py` (repeat = e_diff==0 on all 3 radars). Median share of each repeat spacing, per segment:

| Segment | Spacing 2 | Spacing 3 | Spacing 4 | Repeat rate |
|---|---|---|---|---|
| Fall recording (val) | 0.41 | **0.025** | 0.35 | 0.218 |
| ADL (val) | 0.22 | 0.245 | 0.09 | 0.247 |
| Autre/Empty (val) | 0.31 | 0.275 | 0.28 | 0.259 |

Spacing-3 share as a fall-recording detector:

| Comparison | Train AUC | Val AUC | Note |
|---|---|---|---|
| Whole segment, fall vs Autre/Empty | **0.997** | **1.000** | val n=6 negatives; reproduces L2 and V2 |
| Whole segment, fall vs ADL | 0.70 | 0.73 | |
| 4 s blocks, fall vs Autre/Empty | 0.77 | 0.79 | |
| 4 s blocks, fall vs ADL | 0.77 | 0.78 | |

- L1's round-1 cue (the repeat rate) gives only 0.78 / 0.815 and is the weaker statistic. **L1 concedes that spacing-3 is the right feature.**
- The cue is near-perfect only when aggregated over a whole junction-to-junction segment (~12 s). It needs the junction detector (D-a) to define segments, and a local 4 s window gives only ~0.78.

### D-d. In-fall drop in the repeat rate: useful feature (L1) or weak (L2)? (L2 right; L1 concedes)
`r2.py`, inside the fall vs random same-length windows in the same clip:
- Medians: 0.136 vs 0.240 overall; 0.171 vs 0.233 for evaluable falls.
- But only **60-69% of events** have inside < outside (evaluable: 0.60).
- It is significant at the population level and weak per event. Use it as a minor auxiliary feature only, not for onset detection.

### D-e. Planning baseline (minor; agreed)
- L1 wrote "0.86-0.88". L2 wrote "~0.86" and rejected V8's "0.89 ± 0.03" because it comes from leaky leave-stream-out runs.
- Agreed: plan against **~0.86** (rotation mean 0.861; val 0.884 is the most favourable fold). Treat val differences below ~0.05 as noise unless a paired bootstrap shows otherwise.

## 3. Consolidated new findings (confirmed by code from at least one lead)

| # | Finding | Evidence (lead code) | Implication |
|---|---|---|---|
| N1 | **Sample-and-hold repeats:** ~23-27% of 50 Hz frames repeat the previous frame bit-exactly on all radars. | L1 b_feats / r2; L2 raw check, 26.8% on a raw val ADL chunk. | e_diff has structural zeros. Compute motion, phase and Doppler features from differences between distinct frames, or divide by the frame gap. |
| N2 | **Repeat spacing identifies the corpus.** Spacing-3 share is ~0.025 in fall recordings vs ~0.25-0.28 in ADL and Autre/Empty. | D-c above: segment AUC fall vs Autre/Empty 0.997/1.000, vs ADL 0.70/0.73; 4 s blocks ~0.78. | The strongest cue against the hardest negatives (Autre/Empty), applied as a per-segment aggregate. Pipeline artefact (align.py): ablate it and keep a fallback without it. |
| N3 | **ADL content is bit-exactly reused within a split** (97.3% of val ADL frames). Fall and Autre/Empty frames never are. | L1 c_hash on raw val; L2 composition (80.9%/77.9% cross-stream) and raw lag check. Test uses ADL_USES=2, like val. | A duplicate-content mask on test would mark almost all ADL time as certain negatives. This is **transductive use of a generator artefact: the team lead must rule on it.** It also makes stream-grouped OOF leaky. |
| N4 | **Train OOF is leaky; val is clean.** 90/90 train ADL files sit in more than one stream; each train subject appears in 9-12 of 13 streams; train and val share 0 subjects and 0 ADL files. | L1 a_csv, L2 adl_reuse. | Group CV by subject index and ADL source file, or use split-level holdouts. Never tune post-processing on val. |
| N5 | **Junction-start falls and hard cuts are ignore zones.** All 24 zero-crossfade junctions lead into fall clips and all 24 of those falls are flagged; all 27 falls with source_t_start < 0.25 s are needs_review. Evaluable falls have onset ≥ junction + 1.0 s and duration ≥ 1.0 s. | L1 a_csv, L2 csv_checks; matches `annotate.py` ONSET_VISIBLE_S = MIN_EVENT_S = 1.0. | Drop these as training positives. Post-process with onset ≥ junction + 1 s and duration ≥ 1 s. A prediction after a hard cut costs nothing (ignore zone ±2 s). |
| N6 | **GT is produced by the package's `annotate.propose_fall`, and flags are deterministic.** | L2 re-run on full source recordings: start error median 0.02 s, end error 0.04 s, quality agreement 100%, needs_review 87.9%. L1 read the code. **The crop-only replay** (V7: tIoU≥0.5 for 86.6% of evaluable falls) **was re-run by neither lead.** | The strongest candidate for the boundary regressor inside detected fall segments (theta 0.5 is the weakest term), and for predicting test ignore zones. Verify the crop-level replay first. |
| N7 | **Duration spike at exactly 2.00 s:** 35/478 events (next most common duration: 12); D2 24, Chute04 14; 11 of the 67 evaluable val falls. | L1 a_csv, L2 csv_checks. | Mixture duration prior (a point mass at 2.0 s), validated on leak-free OOF. |
| N8 | **Variance-normalised junction statistic** beats the prototype's relative jump. | D-a above. | Replace jump_max in the structure features, which carry the prototype's only significant gain (+0.086). |
| N9 | **Chute04 is the boundary bottleneck:** val hit@0.5 0.43 and @0.7 0.07, vs 0.80-1.00 and 0.57-0.70 for other types; 4 of the 5 misses @0.3 and 8 of the 14 @0.5. | L1 and L2 best-IoU on val_full.csv. | Type-aware boundary modelling; report per-type hit@0.5 in CV. |
| N10 | **Boundary padding:** a train-tuned shift of -0.3 / +0.2 s gives val 0.904 (F1@0.5 0.830) but F1@0.7 0.311 and median start error 0.40 s. A shift of -0.1 / +0.2 s gives 0.894 with start error 0.24 s. | Official evaluator (L1, L2). Paired CI [-0.005, +0.046]: **not significant**. | Make padding a post-processing hyperparameter tuned on grouped CV, and watch the start-error tie-break. |
| N11 | **D3 is visible on radar 0 and is subject-dependent.** Unflagged D3 (P3_3) has static AUC ~0.965 and is localised on R29171. | F5, F6, F8 above. | Keep radar-0 far-bin features. A test D3 subject like P3_3 would produce evaluable falls. Monitor with leave-P3_3-out CV. |

**Unresolved** (re-run by neither lead, so treat as hypotheses):
- V3: "annotated end is not physical; motion continues ~0.7-1 s past t_end".
- V3: in-cell rates for the mat-cell gate.
- V6: 10x subject variation and "val subjects are the easiest".
- V7: the public seed does not reproduce the composition (irrelevant to the model).

**Rejected:**
- V8: "test ≈ 0.89 ± 0.03". It rests on leaky leave-stream-out runs.
- V4: "+0.26 s widening → 0.909". The shift was chosen with val in view.

## 4. Remaining L1/L2 disagreements
None of substance. All four numeric differences were resolved by code in round 2:
- **D-a:** L2 right.
- **D-b:** both right; the peak bin depends on the definition.
- **D-c:** L2 right for whole segments; L1's time-scale nuance added.
- **D-d:** L2 right.

**Shared caveat for the team lead.** N2, N3, N5 and N6 are generator or pipeline artefacts. They will probably transfer, because test uses the same align.py and ADL_USES=2, but test was built by an unpublished `rebuild_test.py`. Whether N3, a transductive duplicate mask on test streams, is acceptable is a rules decision for the lead. Every submission candidate needs an ablation without these cues.

## L2 response

**L2 signs off on the consolidated verdict.** F1-F10, N1-N11, the rejections and the unresolved list all match L2's round-1 evidence. Any number differences are differences of definition, and none changes a decision. I re-checked three points with code in round 2. Scripts: `/home/user/data_raw/agents/L2/r2_l2.py`, `r2_l2_d3.py` and `r2_l2_d3b.py`, run on cached features for all 18 streams.

1. **D-a: recall on fall-bounding junctions (L1's figure, not previously re-run by L2).**
   - Setup: the variance-normalised statistic, peaks with `distance=10`, matched at ±1.5 s, with the threshold set at train quantiles. Recall on val by junction type:

     | Train quantile | into fall (n=119) | out of fall (n=100) | ADL→ADL (n=355) |
     |---|---|---|---|
     | q0.95 | 0.95 | 0.99 | 0.85 |
     | q0.97 | 0.94 | 0.96 | 0.80 |
     | q0.98 | 0.92 | 0.94 | 0.73 |
     | q0.99 | 0.80 | 0.76 | 0.51 |

   - This confirms that "fall-bounding recall 0.89-1.00, misses are mostly ADL→ADL" holds at the operating thresholds (q0.95-0.98).
   - Caveat: recall drops to ~0.8 at q0.99, so the threshold must be tuned for fall-junction recall, not for overall junction F1.
2. **D-b: D3 "bimodal by subject" (minor, non-blocking).** `r2_f5.py` prints only the pooled values, so I recomputed per subject with L1's definition (complex frame-difference on radar 0, per-bin ADL reference). Median of the normalised-mean argmax per subject:

   | Subject | Median bin | Note |
   |---|---|---|
   | P3_1 | 42.5 | |
   | P3_2 | 48 | |
   | P3_5 | 44 | |
   | P3_9 | 43 | |
   | P3_3 | 61 | the only evaluable D3 subject |
   | P3_8 | 61 | |
   | P3_6 | 53.5 | |
   | P3_7 | 53.5 | |

   - The subject grouping roughly holds. However, the within-subject IQRs are wide, for example P3_3 42.75-62 and P3_7 39.75-62, so "bimodal by subject" overstates it.
   - With the cached `mot_map` instead, P3_5, P3_6 and P3_7 peak at near bins 12-14 and P3_9 at 64 (IQR up to 67.5). These are all 100%-flagged subjects.
   - **Practical rule, slightly widened:** on R29171 use bins ~40-68, and do not treat bins 40-64 as an exclusive D3 cue. P3_3, the only evaluable pattern, sits at ~61-62 under both definitions.
3. **N5: ignore-zone semantics, read in `evaluate.py` `_match_stream`.**
   - Matching to evaluable events comes first, greedily in score order with tIoU ≥ θ.
   - An *unmatched* prediction that overlaps any ignored event widened by ±2 s (`IGNORE_MARGIN_S`) is counted as `ignored_predictions`. It is not a false positive and is excluded from false alarms per hour (the tie-break).
   - So "free" holds only for predictions that overlap a flagged event's [t_start-2, t_end+2]. It does not hold for an arbitrary span after a hard cut that lies outside that window.
   - Consequence: suppressing detections near flagged/D3 events gains nothing. Spending effort on them gains nothing either.

**L2 objections remaining:** none material.
- Housekeeping: the F7 row of the section-1 table omits the spacing-3 sentence that appears in the JSON verdict. It is covered by N2 and D-c.

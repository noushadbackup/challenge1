# Data team, round 1: L2 adjudication (principal data engineer, skeptical justifier)

Scope: I re-checked the ten lead findings (F1-F10) against the 8 analysts' votes, and checked the analysts' new findings. Where votes disagreed or the evidence was thin, I ran my own code. Scripts and outputs are in `/home/user/data_raw/agents/L2/`:

| Script | What it checks |
|---|---|
| `csv_checks.py` | CSV-level checks: F2, F8, N1 (falls at a junction / hard cuts), N3 (2.00 s spike) |
| `adl_reuse.py` | ADL reuse and how subjects spread over streams |
| `raw_dup.py` | Raw h5 test of whether reused ADL content is bit-identical |
| `feat_checks.py`, `more.py` | Cached features: F3, F5, F6, F7, frame repeats |
| `f4.py` | Junction detectors |
| `replay_annot_L2.py` | Re-run of V7's annotator replay |
| `shift_*.csv(.json)` | Boundary-shift checks scored with the official evaluator |

Every number marked **[L2]** comes from code I ran in this session. Numbers marked (Vn) are the analysts' own and were not re-run by me.

## Verdicts on F1-F10

| ID | Verdict | Short reason |
|---|---|---|
| F1 | **confirmed** | Three independent offset scans run by analysts (V1, V2, V6) all reach their extreme at d=184 |
| F2 | **confirmed** (one footnote) | Every number reproduced [L2]. The "min inter-fall gap 7.4 s" in numbers.json is start-to-start; end-to-start is 5.9 s |
| F3 | **corrected** | Qualitatively right; "84x" and "~8 s" are not reproduced with an ADL reference |
| F4 | **corrected** | The AUC holds (0.977-0.989 [L2]); the P/R depends on the detector, and the variance-normalised statistic is better |
| F5 | **corrected** | D3 is NOT weak everywhere: it is seen by R29171 (radar 0) at far bins |
| F6 | **corrected** | Static-change AUCs hold for D1/D2; D3 0.76 is not reproduced; "peak ~1.5 s before onset" is a junction/EMA artefact |
| F7 | **corrected** | Fall-rec vs ADL is stronger (~0.955); fall-rec vs Autre/Empty is not chance (~0.69, noisy) and is near-perfect with frame-repeat timing |
| F8 | **confirmed** (nuance) | Counts exact [L2]; D3 flagging is subject-dependent (P3_3 only 20% flagged) |
| F9 | **confirmed** | Generator does one `rng.shuffle` over all segments; permutation tests by V5/V7 |
| F10 | **confirmed with caveats** | 0.8840 reproduced exactly [L2]; no label leakage into val; train-OOF tuning is optimistic; the realistic level is ~0.86-0.89 |

### F1: I/Q layout (confirmed)
- V1, V2 and V6 each ran a different offset scan on raw h5: Doppler sideband asymmetry, non-circularity and improperness. All three find the extreme exactly at the (k, k+184) pairing:
  - V6 improperness: 0.335 at d=184, 0.40-0.42 at d=183/185, about 0.99 elsewhere.
  - V1 asymmetry: 0.451 at d=184, 0.143 for the interleaved pairing.
- These are three methods and three scripts, with no counter-evidence. The 5 "agree" votes without code add nothing.
- I did not re-run this myself. I accept it on the strength of the convergent code evidence.

### F2: stream structure (confirmed)
Reproduced exactly by `csv_checks.py` [L2]:

| Quantity | Value |
|---|---|
| pre-margin | 0-7.86 s, median 3.03 |
| post-margin | 3.26-10.0 s, median 7.56 |
| fall segment | 7.32-18.64 s |
| other segments | 77-120 s (n=24) |
| fall duration median | 1.74 s |
| fall rate | 58.08 falls/h over 8.231 h |

Additions and corrections:
- The minimum end-to-start gap between falls is **5.9 s**. The 7.4 s in numbers.json is start-to-start (I verified both).
- Evaluable falls have pre-margin ≥ **1.00 s** (p5 1.34, median 3.20) [L2], in agreement with V3, V4 and V7.

### F3: EMA junction spike (corrected)
- My measure [L2]: peak of 3-radar e_mot within 2 s after each junction, divided by the stream's median ADL e_mot at ≥5 s from any junction. Over 2328 junctions:
  - median **54.7x** (p10 11.6, p90 113.8);
  - 41.3x against the all-frame far median.
- This matches V2 (54x) and V4 (63.5x, raw h5). No analyst reproduced 84x, so the multiple depends on the reference.
- Decay to below 3x the ADL median: median **5.1 s**, p90 10.6 s [L2] (V2 and V4 measure the same). The "~8 s" is roughly a p75-p90 figure, not a typical one.
- "90% of energy-baseline FAs start within 1 s of a junction" was reproduced independently with code by V2 and V4 (90.0%, median +0.14 s). I did not re-run it, but I accept it.
- Corrected statement: "junction spikes of median ~55-65x the ADL level (definition-dependent), decaying in ~5 s (p90 ~10 s); 90% of baseline FAs start ≤1 s after a junction."

### F4: junction detectability (corrected)
`f4.py` [L2], two-sided 1 s windows on complex prof_02, threshold tuned on train, event matching at ±1.5 s:

| Detector | Train AUC | Val AUC | Val P / R / F1 |
|---|---|---|---|
| Variance-normalised | 0.995 | **0.989** | **0.961 / 0.799 / 0.873** |
| Relative jump (prototype-style) | 0.990 | 0.977 | 0.861 / 0.710 / 0.778 |

- V2 (0.993; P/R 0.971/0.873) and V5 (0.974; 0.922/0.811) agree.
- V1's 0.855 comes from a weaker positive/negative sampling. It does not contradict the finding.
- The quoted "R 0.82 / P 0.83" is not reproduced for any single detector.
- Corrected statement: "AUC ≈0.98-0.99; the variance-normalised detector reaches val P≈0.96, R≈0.80-0.87. Recall is lowest on ADL→ADL junctions (V2 0.89, V5 0.72), which matter least."

### F5: radar per mat position (corrected)
`feat_checks.py` [L2]: complex 0.2 s block-difference motion inside the event, divided by each radar's ADL median, taking the dominant radar.

| Dataset | Dominant radar (all) | Unflagged | Median ratio (r0 / r1 / r2) | Peak bin median (IQR) |
|---|---|---|---|---|
| D1 | R29444 (idx 1) for 155/160 | 122/122 | 2.1 / **335** / 1.9 | 8.5 (5-14) |
| D2 | R30722 (idx 2) for 154/159 | 115/115 | 15 / 5 / **577** | 11 (7-17.5) |
| D3 | **R29171 (idx 0) for 158/159** | **16/16** | **20** / 0.1 / 0.3 | **61 (46-61)** |

- "D3 weak on all radars" is wrong. D3 is clearly visible on radar 0 at far bins (about 2.4-3.1 m).
- This agrees with V3, V6 and V8, the three code-verified "partially" votes.

### F6: static change and motion-peak timing (corrected)
- **Static change.** I compared the median |z| profile at [t0-2, t0-1] with [t1+1, t1+2], against 4433 ADL pairs with the same spacing inside one ADL chunk [L2]:
  - D1: 0.983 (val 0.983)
  - D2: 0.988 (val 0.985)
  - D3, all falls: **0.595**
  - D3, unflagged only: **0.966** (n=16)
- The lead's D3 figure of 0.76 is not reproduced (V3 0.64, V7 0.55, L2 0.60).
- **Motion peak "~1.5 s before onset": rejected as a fall property.** I took the argmax of e_mot (EMA residual) over the whole fall segment, for evaluable D1 / D2 falls [L2]:

  | | D1 | D2 |
  |---|---|---|
  | Peak before onset | 43% | 32% |
  | Peak inside the fall | 18% | 40% |
  | Peak after the end (next junction) | 39% | 28% |

  The peak is driven by the junctions at both ends of the segment.
- With 0.5 s-smoothed e_diff, the peak is inside the fall for **90% / 89%**, median **+0.69 / +1.06 s** after onset, and in [0, 2] s for 86% / 83% [L2]. This matches V3 and V7. The last sentence of F6 is confirmed.

### F7: static fingerprint (corrected)
`more.py` [L2]: log amp_1s, PCA30 + logistic regression, train→val; blocks within 1.5 s of a junction or a fall are excluded.

| Pair | Val AUC | Notes |
|---|---|---|
| Fall-rec margins vs ADL | **0.955** | V4 0.956, V6 0.980; stronger than the 0.91-0.92 reported |
| Fall-rec vs Autre/Empty | **0.686** | V4 0.69, V6 0.74; not chance |

- The val negatives for Autre/Empty are only 6 segments from 4 streams, so the 0.686 estimate is very noisy.
- The frame-repeat timing cue (below) separates fall recordings from Autre/Empty almost perfectly.

### F8: flagged falls (confirmed, nuance)
- Crosstab [L2]: val D1 7/40, D2 6/40, D3 39/39 flagged; train D3 104/120. Evaluable val = 33 + 34 = 67. The `ignore` column is False for all 478 events.
- Nuance: D3 flagging is per subject. Test_P3_3 is 20% flagged, while the other 7 D3 subjects are 100% flagged [L2].
- The low-quality flag is algorithmic: `annotate.QUALITY_RESIDUAL_GATE = 5.12e-6`. My re-run of the annotator reproduces `quality` for 100% of events [L2].
- A test D3 subject like P3_3 would contribute evaluable D3 falls. Keep D3 detection capability (radar 0, far bins).

### F9: activity before a fall (confirmed)
- `streams.finish_plan` does `segments = falls + others + adl; rng.shuffle(segments)` and then packs the result into streams (I read the code).
- V5 (within-stream permutation, p=0.485) and V7 (TV permutation, p=0.40) confirm it with code.

### F10: prototype score (confirmed with caveats)
- The official evaluator on `reports/prototype/val_full.csv` gives **0.8840**: F1 0.948 / 0.919 / 0.785 / 0.504; TP/FP/FN 62/6/5 at 0.3; median start error 0.22 s [L2].
- **No leakage into val:**
  - The PCA fingerprint is fitted on train only (`add_fingerprint(build(tr_ids), None)`, then applied to val).
  - Labels come only from `frame_labels`.
  - Val shares 0 fall subjects and 0 ADL files with train [L2].
- **Caveat 1, the train OOF is optimistic.** It is used to choose the post-processing.
  - 80.9% of train ADL frames are reused in another train stream (84.5% within the split) [L2].
  - Every train subject appears in 9-12 of the 13 streams [L2].
  - GroupKFold by stream therefore leaks ADL content and subjects (V4, V8).
- **Caveat 2, noise.**
  - Bootstrap CI over 5-min windows: [0.82, 0.94] (V1, V5).
  - Rotating val with train stream sets gives a mean of 0.861, with val the most favourable fold (V1).
  - Read 0.884 as "~0.86-0.89", not as a precise estimate.

## Analysts' new findings: kept (confirmed by L2 code)

1. **Sample-and-hold frame repeats (V1, V2, V8).**
   - 26.8% of consecutive frames in a raw val ADL chunk are bit-identical on all 3 radars [L2].
   - From cached e_diff (repeat on all radars): ADL 0.238, fall recordings 0.214, Autre 0.25, Empty 0.24-0.26.
   - Inside falls the rate is 0.172, against 0.233 for random same-length windows in the same clip (Wilcoxon p=5.9e-13, n=477). Per event it is weak: only 63% of falls have inside < outside [L2].
   - Implication: e_diff has ~25% structural zeros, so frame-level differences should use distinct frames.
2. **Repeat spacing fingerprints the source corpus (V2).** Spacing-3 share among repeat intervals, per segment [L2]:

   | Segment type | Spacing-3 share |
   |---|---|
   | Fall recordings | 3.2-3.5% |
   | ADL | 27-29% |
   | Autre/Empty | 27% |

   - Segment-level AUC, fall-rec vs Autre/Empty: **0.997 train / 1.000 val** (val has only 6 other segments).
   - Fall-rec vs ADL: 0.70 / 0.73.
   - This is the missing cue against the hardest negatives. It is a pipeline artefact (align.py); the test set uses the same aligner but an unpublished rebuild script, so ablate it and keep a fallback.
3. **ADL content is bit-exactly reused within a split; fall and Autre/Empty content never is (V4, V7, V8).**
   - By composition: train 80.9% of ADL frames in another stream; val 77.9% in another stream and 100% somewhere else in the split. Each of the 359 + 119 fall segments and 18 + 6 other segments has a unique path [L2].
   - Raw check: a 583-frame overlap of `Faire_the_0` in val_00 and val_01 is 100% bit-identical at lag 1 [L2].
   - Test uses `ADL_USES["test"] = 2`, the same as val.
   - Using this on unlabeled test streams is transductive. The team lead must rule on it.
4. **Train OOF leakage (V4, V8).** Confirmed by the numbers in F10, caveat 1. Model selection needs grouping by subject and ADL file, or a split-level holdout.
5. **Junction-start falls are free zones (V1, V4).**
   - 27 falls have source_t_start < 0.25 s; all are needs_review, so ignored. 24 have pre-margin < 0.05 s, all flagged.
   - All 24 zero-crossfade junctions lead into fall clips (previous segment: ADL 19, fall 5). Every other junction has 0.2 s overlap (2302), apart from 2 partial ones [L2].
   - Evaluable falls have onset ≥ 1.0 s after the junction and duration ≥ 1.0 s (44 non-evaluable falls are shorter) [L2].
6. **2.00 s duration spike (V1).**
   - 35 events are exactly 2.00 s; the next most common duration (1.06 s) has 12. They split D2 24 / D1 8 / D3 3 and Chute04 14.
   - 27 are evaluable, including 11 of the 67 evaluable val falls [L2].
7. **GT is produced by the package's own `annotate.propose_fall` (V7).**
   - Re-running it on the full source recordings [L2]: |start error| median 0.02 s, 72.8% < 0.1 s, 84.1% < 0.5 s; |end error| median 0.04 s; quality agreement 100%; needs_review agreement 87.9%.
   - The README (line 64) states the same rule.
   - Not re-run: V7's crop-only replay on stream-visible segments (tIoU ≥ 0.5 for 86.6%). That is what matters on test. Verify it before relying on it.
8. **The variance-normalised junction statistic beats relative jump (V2, V5).** See F4: val AUC 0.989 vs 0.977, P/R 0.961/0.799 vs 0.861/0.710 [L2].
9. **Boundary padding helps the mean F1 at 0.1/0.3/0.5 (V4, V5, V8).** Applying the train-tuned shifts to val_full [L2]:

   | Shift (start / end) | Ranking F1 | F1 @0.5 | F1 @0.7 | Median start error |
   |---|---|---|---|---|
   | none | 0.8840 | 0.785 | 0.504 | 0.22 s |
   | -0.3 / +0.2 (V5) | 0.9037 | 0.830 | 0.311 | 0.40 s |
   | -0.1 / +0.2 (V8) | 0.8938 | 0.800 | 0.400 | 0.24 s |

   - This is within noise: V5's paired CI is [-0.005, +0.045].
   - Make the shift a post-processing hyperparameter tuned on properly grouped CV, and keep the start-error tie-break in view (the -0.1 start variant barely hurts it).
10. **Chute04 is the boundary bottleneck (V6, V8).**
    - Best-IoU hits on val [L2]: hit@0.5 = 0.43 and hit@0.7 = 0.07 for Chute04, against 0.80-1.00 and 0.57-0.70 for the other types.
    - Chute04 accounts for 4 of the 5 misses at 0.3 and 8 of the 14 at 0.5.

## Analysts' new findings: not kept (unverified by L2, or overstated)

- **V3, "annotated end not physical":** plausible, since the labels come from a static-change algorithm (item 7), but I did not re-run it. Unresolved.
- **V3, mat-cell gate:** consistent with my F5 table, but the in-cell rates for hard negatives were not re-run. Unresolved.
- **V6, 10x subject variation and "val subjects easiest":** not re-run. Unresolved.
- **V7, public seed does not reproduce the composition:** not re-run. It is irrelevant to the model.
- **V5, "val cannot resolve gains below ~0.05":** two independent bootstraps (V1, V5) agree on a CI of about ±0.06. I accept it as methodology, but did not re-run it.
- **V8, "test score about 0.89 ± 0.03": rejected as optimistic.** It rests on leave-stream-out runs that include the ADL/subject leakage. V1's rotation gives a mean of 0.861. Use ~0.86 as the planning baseline.
- **V4, "+0.26 s widening gives 0.909":** the shift was chosen with val in view, so it is not a valid estimate. The train-tuned version is item 9.

## Skeptic's notes for the plan
- Several of the strongest new cues are generator or pipeline artefacts: repeat spacing, ADL bit-duplication, hard cuts into falls, and the annotator replay. They will likely transfer, because test uses the same align.py and ADL_USES=2, but test was built by an unpublished `rebuild_test.py`. Every submission candidate needs an ablation without them and a physics-only fallback.
- All model and post-processing selection should move to CV grouped by subject index and ADL source file (or a split-level holdout). Stream-grouped OOF is leaky.

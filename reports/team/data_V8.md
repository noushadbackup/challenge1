# V8: ML engineer / evaluation reproducer (official scorer, robustness to the hidden test)

All scratch code is in `/home/user/data_raw/agents/V8/`. I did not edit any lead script or the package. Every number below comes from code I ran in this session.

## What I ran

| script | purpose |
|---|---|
| `myscore.py` | **Independent re-implementation of the event scorer**, written from the README rules: greedy one-to-one matching by score, tIoU >= theta, ±2 s ignore zones, `--ignore-flagged`. |
| `f10_rescore.py` | Re-scores the lead's `val_{physics,structure,full}.csv` with my scorer, then compares the result with the official JSON. Also runs a stream bootstrap and an error decomposition. |
| `f10_cv.py` | Retrains the prototype from the lead's feature builders with my own driver, checks that 0.884 comes out exactly, then runs a **leave-one-stream-out CV over all 18 streams** with the post-processing fixed at (0.4, 3, 0.8, 0.6). |
| `lso_analysis.py` | Scores each stream separately, simulates 5-stream test sets, and breaks recall and IoU down by fall type and by dataset. |
| `f2_check.py` | Recomputes the fall timing from the CSVs. Each event is joined to the composition segment that contains it, not matched by id, and the result is cross-checked against `source_t0/source_t_start`. |
| `f5_check.py` | Re-derives the mat-to-radar mapping with a **different signal** from the lead's EMA residual: the block-to-block change of the complex 0.2 s profile, `abs(diff(z))`. That change is ratioed to the per-bin ADL median of the same stream. I also report which radar wins for each event. |
| `reuse.py`, `dup.py`, `dupframes2.py`, `dupkind.py`, `dupfilter.py` | Measure how much source material is reused across streams. Find exact duplicate raw frames by hashing every val frame of the raw h5. Measure the effect on predictions. |
| `fpfn.py`, `shift_tune.py` | Audit the remaining val errors. Tune a start/end shift on the **train** LSO predictions only, then apply it once to val. |

## Votes

| # | vote | conf | by code | evidence |
|---|---|---|---|---|
| F1 | agree | 0.8 | no | I did not test it directly. My F5 check builds complex `z = x[:184] + 1j*x[184:]`. It gives sharp, consistent per-bin peaks (D1 → radar 1 bin 13, D2 → radar 2 bin 11), which fits the I/Q reading. |
| F2 | **agree** | 0.95 | **yes** | All 478 events sit in exactly one fall segment with the matching path. Pre-margin is median 3.03 (0–7.86) s, post-margin 7.56 (3.26–10.0) s, segment 12.4 (7.32–18.64) s, duration 1.74 (0.48–5.0) s. Autre/Empty run 77–120 s, and the rate is 58.08 falls/h. All of this matches the lead. **Extra detail:** 409/478 (86 %) fall segments start at `source_t0 = 0`, so the configured 3–8 s pre-margin is clipped by the start of the recording. 49 % of falls start less than 3 s after the junction, 53 (11 %) start less than 1 s after, and 24 start right at it. |
| F3 | agree | 0.7 | no | Not re-run. It fits the baseline: 44 of its 140 val predictions fall in duplicated ADL content (see N1), and removing them cuts FP from 100 to 57. |
| F4 | agree | 0.7 | no | Not re-run. The prototype's `jump_max` structure features add +0.086 on val (physics 0.790 → structure 0.876), which shows junction information is usable. |
| F5 | **partially** | 0.85 | **yes** | D1 and D2 are confirmed and even stronger than reported. With a different signal, R29444 wins **157/160** D1 events (122/122 unflagged, 40/40 val) at a peak of ×211, bin 13. R30722 wins **159/159** D2 events at ×408, bin 11. **D3 is not "weak on all radars"**: R29171 wins 149/159 D3 events (16/16 unflagged, 39/39 val) at bin ≈62 (about 3.2 m), median ×16.6 (×59 on unflagged). D3 is far from the radars, not invisible. |
| F6 | agree | 0.7 | no | Not re-run. It fits my error audit: the remaining misses are localization errors, not missing signal. |
| F7 | agree | 0.75 | no | Not re-run. N1 below is a stronger, parameter-free version of the same effect for ADL. |
| F8 | **agree** | 0.95 | **yes** (counts) | My scorer finds 67 evaluable val falls (D1 33, D2 34, D3 0). The full model's predictions sitting on ignore zones (17 of them) are dropped at no cost. |
| F9 | agree | 0.7 | no | Not re-run. The generator (`build_streams.py`) shuffles segments, so there is no causal activity context. |
| F10 | **agree** | 0.85 | **yes** | (1) My scorer reproduces the official numbers to 4 dp: 0.7903 / 0.8759 / 0.8840; 0.948/0.919/0.785/0.504; 62/6/5; median start error 0.22 s. (2) Retraining with my own driver gives val 0.884 again, so the result is deterministic. (3) Over the 5 val streams, a bootstrap gives a 90 % CI of **[0.865, 0.912]**. (4) LSO over 18 streams gives train 0.880. Random 5-stream subsets of the 18 per-stream results give mean 0.893, sd 0.026, 5–95 % **[0.85, 0.94]**, which is a realistic band for the 5 test streams. (5) Val labels never enter the pipeline: the PCA is fit on train, and per-stream thresholds are unsupervised. **Caveat:** the train OOF that tunes post-processing is optimistic because of within-split reuse (N1), so 0.901 OOF is not a fair number. |

## New findings

### N1. Within a split, 81 % of ADL content is an exact copy of ADL in another stream. Fall, Autre and Empty content is never repeated.

* Composition: ADL reuse factor is 1.93 (train) and 2.00 (val). Every ADL source file appears in more than one stream. On average 81 % of a stream's ADL seconds also appear in another stream of the same split. The config says `adl_uses: test: 2`, so **the test split is built the same way.** Fall segments (`fall_uses: 1`) and Autre/Empty segments are all unique.
* Raw signal check: I hashed every frame of the 5 raw val h5 files and ignored repeats closer than 2 s, which are sample-and-hold copies. Exact far-duplicates cover **97.3 % of ADL frames and 0.0 % of fall-recording, Autre, Empty and fall-event frames.**
* What it changes:
  1. **CV is leaky.** GroupKFold by stream puts the same ADL frames, and the same fall subjects (18/18 train subjects span more than one stream), in both the training and the held-out folds. OOF scores and OOF-tuned thresholds come out optimistic. Val LSO, with the other 4 val streams in training, scores 0.928, while the train-only model scores 0.884 on the same streams. Use a split-level holdout (train → val) for the final decisions.
  2. **A free, near-perfect ADL negative mask, computable from the unlabeled test streams.** It cuts the energy baseline's FP@0.3 from 100 to 57 (rank F1 0.315 → 0.401). The full prototype already makes no FP in ADL (0 affected predictions), but the mask is a safety net in case the fingerprint generalizes worse to test subjects. **Check the rules on transductive use of test data before relying on it.** It is a property of the generator, not of falls.
  3. About 23.5 % of frames repeat the previous frame exactly (sample-and-hold to 50 Hz). Frame-difference features (`e_diff`) are exactly zero on those frames, which adds periodic noise. Use the block or median form, as the prototype does.

### N2. The remaining errors are localization errors, mostly on Chute04 (chair falls).

* Val, full model: 5 of the 6 FP@0.3 sit **inside the same fall segment** as a missed fall, so each one is a double penalty (FP + FN). The sixth sits in an Autre segment. 4 of the 5 FN are **Chute04**, with predictions landing 2.6–5.7 s after the true onset.
* LSO over 18 streams: Chute04 has recall@0.3 of 0.836 and recall@0.5 of 0.727, against 0.92–0.96 and 0.86–0.89 for Chute01–03. Its median IoU is 0.686 against 0.755–0.800. D2 is weaker than D1 (recall@0.3 0.878 vs 0.943).
* What it changes: two things should help. First, keep one event per junction-to-junction fall segment. Second, add a Chute04-aware boundary head, for example a sit-then-slide two-phase static change. The rank-F1 headroom is at θ = 0.5.

### N3. A small boundary shift transfers from train to val. The score has about ±0.04 noise on 5 streams.

* I tuned a constant start/end shift on the **train LSO predictions only**: start −0.1 s, end +0.2 s. Train rank F1 went from 0.880 to 0.889. Applied once to val it gives **0.884 → 0.894** (F1@0.3 0.919 → 0.933, F1@0.5 0.785 → 0.800). That gain is inside noise but costs nothing, so keep it as a post-processing step.
* Test-score expectation: about 0.89 ± 0.03 (sd of 5-stream subsets is 0.026). Differences under about 0.03 on val are not evidence. Decide between models with LSO-by-split or several seeds, not a single val number.
* Format robustness: the official validator rejects `t_end > duration` and non-finite scores. Any shift must clip to `[0, T/50]`, and `shift_tune.py` does.

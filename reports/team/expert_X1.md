# Expert review X1: UWB / impulse-radar signal processing

Reviewer: X1 (X4M200, I/Q, micro-Doppler, clutter removal). Scope: PLAN.md v1, read alongside REPORT.md and DATA_TEAM_VERDICT.md.
**Verdict: sound with changes.** The detection architecture is right. The boundary stage (S4) and the Doppler/phase work package (E3) are aimed at the wrong signal, and one voting rule in §3 contradicts the radar geometry.

All numbers in this file come from scripts I ran in this session. The scripts are in `/home/user/data_raw/agents/X1/`:

- `x1_replay_hybrid.py` and `x1_replay_gate.py`: boundary replay, scored with the official evaluator.
- `x1_phase.py` and `x1_phase2.py`: frame repeats and phase wrapping.

Anything not measured is marked *(assumption)* or *(estimate)*.

---

## 1. The key fact the plan does not use: the ground truth is a signal-processing algorithm we can run

`src/abc2027/annotate.py::propose_fall` generated every `t_start`/`t_end` (data-team finding N7). Here is what it does:

1. **Motion.** `motion = Σ over all 368 real values of (x_t − x_{t−1})²`. This is the cached `e_diff`.
2. **Spike suppression.** Single-radar bursts that last no more than 0.3 s and change state by less than 0.15 are clamped to p20.
3. **Smoothing.** The state and the motion are smoothed with a 0.4 s box filter.
4. **State.** For each radar, `state = d_before / (d_before + d_after)`, where d is the squared distance to a reference profile:
   - the "standing" profile is the median of the first 1 s;
   - the "floor" profile is the median of the last 1.5 s of the recording.
5. **Anchor radar.** The radar with the largest `‖after − before‖²`.
6. **Transition window.** Over 1.5 s windows, `score = rise · log1p(motion_sum / median)`, eligible only where the rise is at least 0.12. The bounds are the 20–80 % crossings of that rise.
7. **Expansion.** The bounds are widened while motion stays at or above `p30 + 0.15·(peak − p30)`, by at most 3 s to the left and 2 s to the right.

From a radar point of view, this is a junction-relative two-profile clutter model with a frame-difference motion detector. ruptures PELT, or a generic offset regressor on 0.2 s blocks (plan S4), only approximates it. We can run it directly on each fall-recording segment.

### Measured: annotator replay as boundary refiner (val, official evaluator, `--ignore-flagged`)

Method:

- I took the prototype's predictions (`reports/prototype/val_full.csv`).
- For every prediction whose midpoint lies in a fall-recording segment, I re-ran `propose_fall` on the **stream crop**, read from the val h5 (crossfade included). The segment edges come from `streams_composition.csv`, jittered by ±J s and trimmed inward by TR s.
- I replaced the bounds only if the replayed interval overlaps the original (tIoU > G).
- Scores and detections were kept unchanged.

| variant | rank F1 | F1 @0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | median \|start err\| | mean tIoU |
|---|---|---|---|---|---|
| prototype (re-scored) | 0.8840 | .948/.919/.785/.504 | 62/6/5 | 0.22 s | 0.698 |
| replay, no gate, exact edges | 0.8841 | .928/.899/.826/.667 | 62/9/5 | 0.04 s | 0.814 |
| replay, no gate, ±1.0 s jitter, no trim | **0.7146** | .763/.734/.647/.518 | 51/21/16 | 0.08 s | 0.782 |
| replay + gate G=0, exact edges | 0.9037 | .948/.919/.844/.667 | 62/6/5 | **0.04 s** | 0.808 |
| replay + gate G=0, ±0.5 s jitter, 0.5 s trim | 0.9037 | .933/.919/.859/.696 | 62/6/5 | 0.04 s | 0.817 |
| replay + gate G=0.3, ±1.0 s jitter, 1.0 s trim | **0.9086** | .933/.919/.874/.652 | 62/6/5 | 0.04 s | 0.808 |

Best tIoU per fall type, comparing the prototype with the replay (±0.5 s jitter, 0.5 s trim, G=0.1):

| type | n | prototype hit@0.5 / @0.7 | replay hit@0.5 / @0.7 |
|---|---|---|---|
| Chute01 | 14 | 0.86 / 0.57 | 0.71 / 0.64 |
| Chute02 | 19 | 1.00 / 0.58 | 1.00 / 0.68 |
| Chute03 | 20 | 0.80 / 0.70 | 0.95 / 0.85 |
| **Chute04** | 14 | **0.43 / 0.07** | **0.71 / 0.57** |

What this shows:

- **The replay fixes the Chute04 boundary bottleneck** flagged by the data team. It also cuts the median start error, which is tie-break 1, by about 5x (0.22 s to 0.04 s).
- **The replay is fragile in one specific way.** If the crop reaches into the neighbouring segment, the "standing" profile is wrong and the replay picks the wrong motion burst. With ±1 s junction error and no trim, the score drops to 0.715. V7's "robust to ±1 s" result jittered inside the *source* recording, where there is no foreign content. On real streams that does not hold.
- **Trimming inward by at least the junction error removes the failure.** Evaluable falls start at least 1 s after the junction.
- **The replay hurts some Chute01 events.** A per-event selector is needed, not a blind replace.
- **Caveats:**
  - Segment identity is still oracle; only the edge positions were jittered.
  - The val set has n=67, so +0.02 to +0.025 is not significant on its own. I did not compute a bootstrap CI.
  - The 5 FN and 6 FP are unchanged, because they are detection errors.

---

## 2. Changes (prioritised)

### P0-1. Replace S4 with "annotator replica + learned selector/corrector"
- **What to build:**
  - Junction detector (variance-normalised jump, threshold chosen for fall-junction recall, per L2's table) → candidate segment `[j_k + TR, j_{k+1} − TR]`.
  - Run `abc2027.annotate.propose_fall` on that crop, wrapping each radar as a `RadarTrack` with `time_s = arange(T)·0.02` and `motion = Σ diff²` (same wrapper as V7 and my script).
  - A LightGBM selector/regressor decides `final = replay | model | blend` and regresses residual offsets. Its inputs:
    - the replay internals (max rise, whether `eligible` was satisfied, anchor radar, motion_sum/median, onset spread across qualified radars, t_peak, t_settle);
    - tIoU(replay, model);
    - segment length;
    - distance from onset to segment start;
    - the model's own bounds.
- **TR:** set it from the measured junction-error distribution, at about its p95 and at least 0.5 s. Do not use 0. Also try a crop that ends at the *next* junction but starts at the previous junction + TR.
- **Fallback:** keep the model's bounds when no junction is found within ±20 s, or when the replay's rise is below 0.12.
- **Expected gain** *(estimate)*:
  - +0.02 to +0.03 ranking F1, from F1@0.5 going 0.785 → 0.84–0.87;
  - median start error 0.22 → about 0.04 s, which secures tie-break 1;
  - F1@0.7 from 0.50 → 0.65–0.70 (not ranked, but it shows boundary quality).
- **Verify:**
  - Grouped CV on train by fall subject, using our own junctions and not the composition CSV.
  - Then the pseudo-test bank, then a single holdout run on val.
  - Report hit@0.5 per Chute type and the replay-failure rate (tIoU(replay, GT) < 0.3).

### P0-2. Junction timing precision becomes a first-class metric
- The replay only works when the crop stays inside the fall segment. Measure the **signed time error of fall-bounding junctions** (p50, p95, max), not only junction P/R at ±1.5 s.
- **Target:** p95 at or below 0.5 s.
- **How:** refine each detected junction to the frame level with the per-frame jump on `prof_02`, then on raw 50 Hz frames inside ±0.5 s, using the 0.2 s crossfade signature. The data team's N2 shows ADL edges can be found with a median error of 0.00 s.
- **Expected gain:** this lets TR shrink to about 0.3 s. It protects falls whose onset is 1.0–1.5 s after the junction, which a 1 s trim would cut into.

### P0-3. Feed the annotator's own signals into S3 (detector), not only S4
- A frame classifier with fixed ±1/±2 s lags cannot express *segment-relative* quantities. For each block, compute these per radar, inside the current junction-to-junction segment:
  - `state` (distance to the segment's first-1 s and last-1.5 s profiles);
  - the 1.5 s `rise`;
  - `score = rise·log1p(motion_sum/median)`;
  - the anchor radar's identity;
  - the argmax position of `score` within the segment.
- Use the raw 368 values as `annotate.py` does, so the features match the GT definition exactly.
- **Expected gain:** fewer Chute04 misses and better score ordering *(estimate: +0.005 to +0.015; unmeasured)*.
- **Verify:** an ablation on grouped CV and the pseudo-test bank.

### P1-1. Re-scope E3 (phase/Doppler): wrap-robust features on clutter-removed, de-duplicated frames
These measurements are on 67 val evaluable falls, using the dominant radar and bins 3–24:

- **Repeats.** 25.3–27.2 % of frames are bit-exact holds on every radar and every val stream. The distinct-frame rate during falls is about 34 Hz (D1) and 45 Hz (D2), not 50 Hz.
  - An STFT or phase-difference computed on the 50 Hz grid mixes true 20 ms steps with 0 ms and 40 ms steps.
  - **Drop the held frames and use the true Δt.** Use `scipy.signal.lombscargle` or resampling if a spectrum is needed.
- **Raw phase is dominated by static clutter.** Without clutter removal, the energy-weighted share of phase steps above π/2 in the fall is only 1.3 % (D1) and 10.5 % (D2). Lag-1 coherence is 0.95/0.70 in the fall against 1.00 before it. So "phase velocity" on raw z measures clutter, not the body.
- **After a 0.34 s centred high-pass** on distinct frames, 10.7 % (D1) and 25.9 % (D2) of the fall energy has |Δφ| > π/2 per frame. If uniform, the share would be 0.5.
- **Aliasing.** Assuming the X4M200 default 7.29 GHz carrier (λ ≈ 4.1 cm, *assumption*), the unambiguous radial speed is λ·f_s/4, about 0.35–0.46 m/s at 34–45 Hz. Fall speeds of 1–3 m/s will alias.
- **Therefore: do not unwrap or integrate phase into displacement or velocity.** Use these instead:
  - **Lag-1 complex coherence** |Σ z_t z*_{t−1}| / Σ|z_t||z_{t−1}| over mat bins (5–21 on R29444/R30722, 40–68 on R29171), at 0.2 s and 1 s.
  - **Circular variance** of Δφ and the **share of high-|Δφ| energy**, which acts as a Doppler-spread proxy.
  - **STFT spread** on high-passed distinct frames: 0.64 s Hann windows, energy share in |f| > f_s/4, and spectral entropy (`scipy.signal.stft`).
  - **Range-centroid velocity:** the motion-weighted bin centroid × 5.14 cm per second. It is not aliased below about 1 bin per frame, roughly 2 m/s.
  - **Range spread:** the number of bins above the motion p90.
- **Clutter removal must reset at junctions.** Use the segment's own median, or a high-pass of 0.5 s or less. Never a 3 s EMA across a junction (F3).
- **Expected gain:** small for detection (+0 to +0.01, *estimate*). The real value is an ensemble member that is decorrelated from the static-profile cues. Keep it P1, not on the critical path.

### P1-2. Replace "≥ 2 radar agents agree" with anchor-radar logic
- Each mat is seen mainly by one radar (F5): D1 by R29444, D2 by R30722, D3 by R29171 at far bins. `annotate.py` itself keeps `SINGLE_RADAR_SHARE = 0.70` and uses one anchor radar.
- A rule that needs two radars to agree would cut D1/D2 recall.
- **Instead:**
  - choose the anchor per segment by the largest static change;
  - pass the anchor's features and the radar index to the model;
  - use the other radars only as context, for example motion elsewhere in the room, which indicates an ADL walker.
- **Verify:** per-dataset recall with and without the rule.

### P1-3. Replicate the motion definition on the stream grid
- The annotator computes motion on *raw* frames before 50 Hz alignment. The stream's `e_diff` puts zero on held frames and a doubled step on the next one.
- V7 attributes the remaining replay error to this difference. Two options:
  - spread each distinct frame's Δ² evenly over its hold length;
  - recompute motion on distinct frames only, then interpolate (`np.interp`, as `_resample` does).
- **Verify:** replay tIoU against GT on train crops, with exact edges, before and after.
- **Expected:** a few points of hit@0.7 *(estimate)*.

### P2-1. Deep models (S3-d, S3-e) should take physically meaningful inputs
- If CNN/CRNN members are built (torchvision `resnet18`, a CRNN in PyTorch), use per-radar 2-channel maps on mat bins only, from distinct frames and reset at junctions:
  - high-passed |z|;
  - lag-1 coherence.
- Do not use the raw 368-value vector, which mixes I and Q as if they were range.
- Low priority: the replay plus the GBM already covers the physics.

### P2-2. Quality-gate replica as a diagnostic
- Compute the strongest-radar residual (mean squared deviation from the segment median) against the 5.12e-6 gate. This predicts `quality=low`, which V7 reproduced for 100 % on full recordings.
- It does not change the score, since predictions in flagged zones ±2 s are free.
- Use it in CV to drop flagged-like segments from training positives and to break down FPs.

### Plan text to amend
- **S4 row:** "ruptures PELT/Binseg + offset regressor" → "annotate.propose_fall replica on junction-trimmed segment + LightGBM selector/residual regressor; ruptures only as a fallback".
- **Risks row** "regress to the annotation rule": make it concrete as above.
- **E3:** "phase velocity" → the wrap-robust coherence/Doppler-spread features above, on distinct frames.
- **§3 voting item 1:** anchor radar instead of "≥ 2 radar agents agree".

## 3. Metrics to add
- Signed time error of fall-bounding junctions (p50/p95/max).
- Replay tIoU against GT and the replay-failure rate (< 0.3), per Chute type.
- Per-type hit@0.5 and hit@0.7.
- Share of start errors within ±0.1 s.
- Selector accuracy (replay vs model) on CV.
- Distinct-frame rate per segment.
- Anchor-radar agreement between replay-on-crop and replay-on-full (train only, for the diagnostic).

## 4. Risks
- **Test labels must come from the same `propose_fall`.** If the organisers hand-corrected any test labels, a pure replay overfits the algorithm. The learned selector, with model bounds as a fallback, hedges this.
- **The replay is very sensitive to junction position** when crops include foreign content (0.715 at ±1 s without trim). Trimming and frame-level refinement are mandatory.
- **Using `annotate.py` at inference.** It is package code, not LIARA data, so I believe it is allowed. The team lead should confirm this together with the other generator-artefact cues.
- **Assumed carrier.** The 7.29 GHz carrier is the X4M200 default and is not verifiable from the package. The aliasing argument holds for either X4 band (7.29 or 8.75 GHz).
- **Small validation set.** With 67 evaluable val falls, the measured +0.02 to +0.025 is within noise. Promote only through CV plus the pseudo-test bank.

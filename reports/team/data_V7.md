# V7: data engineer, reverse-engineering the stream generator and annotations

**Role:** I read the package's generator (`scripts/build_streams.py`, `src/abc2027/streams.py`), its annotator (`src/abc2027/annotate.py`), its aligner (`align.py`) and its evaluator. Then I checked the lead's F6, F8 and F9 with code I wrote myself, using different methods from the lead's. I also looked for exploitable structure in how the data was generated.

All scratch code is in `/home/user/data_raw/agents/V7/`. Every number below comes from running these scripts in this session:

| script | what it does |
|---|---|
| `replay_annot.py` | Runs the package's own `annotate.propose_fall` on all 478 full fall recordings (`segments/*.npy`) and compares the output with the ground truth (GT). |
| `replay_crop.py` | Runs the same annotator only on the cropped fall segment that actually appears in the stream, which is what we will see on test. |
| `replay_jitter.py` | Measures how sensitive that replay is to errors in where the crop (the junction) is placed. |
| `f8_check.py` | Counts flags using the official `load_ground_truth(ignore_flagged=True)` and breaks down the reasons for each flag. |
| `f6_check.py`, `f6_feats.py` | Independent static-change AUC on complex amplitude, with harder negatives. Also tests where the motion peak falls under three different motion definitions. |
| `regen_f9.py` | Re-plans the streams with the public seed. Permutation test of the activity next to each fall. |
| `dup_frames.py`, `dup_segments.py` | Finds frames that are exact duplicates across segments. Recovers segments without using the composition file. |

---

## Votes

| F | vote | conf | by code | evidence |
|---|---|---|---|---|
| F1 I/Q split | agree | 0.70 | no | I did not test it directly. Building the static-change feature on the complex amplitude \|I+jQ\| (bins 0–183 as I, 184–367 as Q) gave AUCs of 0.969 and 0.977, as strong as the lead's. That is consistent with F1 but does not prove it. |
| F2 timing | agree | 0.85 | yes (partly) | The generator draws margins as `before=min(U(3,8), source t_start)` and `after=min(U(5,10), t_last−t_end)`. For the 253 evaluable falls (train+val), the fall starts this long after its junction: quantiles 1.0 / 1.65 / **3.2** / 5.05 / 7.2 s. The minimum is **1.0 s**, because the annotator flags `start_not_visible` when t_start < 1 s. In 85.6 % of fall segments the crop starts at the very first frame of the source recording (the 3–8 s margin is clipped). |
| F3 EMA junction spike | agree | 0.75 | partial | With the cached `e_mot` (3 s EMA residual), the per-segment maximum is inside the fall for **0 %** of evaluable D1 falls and 1 % of D2 falls, at a median of **−2.7 s / −2.4 s** relative to onset. That is exactly the junction position, since onset is about 3 s after the junction. This supports "the junction dominates the EMA energy". I did not re-check the 84× figure or the 90 % false-alarm figure. |
| F4 junction detectability | agree | 0.60 | no | I did not rerun it. See new finding N2: on ADL boundaries, junctions can be found exactly (start error 0.00 s median), so this AUC of 0.97 is a lower bound on what is achievable. |
| F5 mat to radar | agree | 0.70 | no | Plausible, and matches `numbers.json` (D1→R29444 bin 10 ×51, D2→R30722 bin 13 ×108). Not checked independently. |
| **F6** fall vs motion peak, static change | **partially** | 0.80 | **yes** | **Static change: confirmed.** My method (median \|I+jQ\| profile in [t0−2, t0−1] s vs [t1+1, t1+2] s, normalized squared distance, max over radars) gives AUC **0.969 (D1), 0.977 (D2)** against ADL pairs with the same gap. Against harder negatives it is weaker: **0.875 / 0.911 against Autre/Empty**, and 0.94 / 0.96 against windows taken inside the pre-fall margin. D3 over all falls is **0.55** (the lead had 0.76), but on the 16 evaluable D3 falls it is 0.977. **"The motion peak is ~1.5 s before onset": refuted as a property of the fall.** It is an artifact of measuring motion with the EMA residual. With frame-difference energy (each radar normalized by its median, max over radars, 0.4 s smoothing), the global maximum over the whole fall segment lies **inside** the annotated fall for **93 % (D1) / 88 % (D2)** of evaluable falls, at a median of +0.76 s / +1.06 s after onset. With the cached `e_diff` the figures are 89 % / 88 %. With `e_raw` they are 41 % / 59 %. With `e_mot` they are 0 % / 1 %, with the peak at −2.7 s, which is the junction. The "walking onto the mat" explanation is not supported. The last sentence of F6 (frame-diff peak 0–2 s after onset) is confirmed: 89 % (D1), 84 % (D2) of peaks fall in [0, 2] s. The 11 % / 35 % figures are not in `numbers.json` or in any script, so I could not reproduce them. |
| **F8** D3 flagged, 67 evaluable | **agree** | 0.95 | **yes** | Through the official `load_ground_truth(ignore_flagged=True)`: the ignore set equals `quality==low or needs_review` exactly, and the `ignore` column is false for all rows. Flagged share by split and dataset: train D1 25.8 %, D2 31.9 %, **D3 86.7 %**; val D1 17.5 %, D2 15.0 %, **D3 100 %**. Evaluable val: **33 + 34 + 0 = 67**. Evaluable train: 89 / 81 / 16. **Mechanism:** D3 is flagged because its strongest-radar residual is under the gate 5.12e-6 (`quality=low` for 85 % of train D3 and 100 % of val D3). These flags come from an algorithm, which I reproduce exactly (see N1), so test D3 will almost certainly also be mostly ignore zones. Note that **15–32 % of D1/D2 falls are also ignore zones**, flagged as `needs_review` for these reasons: radar_onset_disagree, start_not_visible, event_under_1s. |
| **F9** no activity context | **agree** | 0.90 | **yes** | I used a different test from the lead's chi-squared: total-variation distance between the labels of ADL chunks next to a fall and all ADL chunks, with a 5,000-sample permutation null. Preceding chunk: n=389, TV=0.067, **p=0.40**. Following chunk: TV=0.074, **p=0.22**. The kind of segment before a fall (adl 82.1 %, fall 17.3 %, other 0.6 %) is close to the base rates (78.6 / 20.4 / 1.0 %). The cause is structural: `finish_plan` does a single `rng.shuffle` over all segments. |
| F10 prototype 0.884 | agree | 0.55 | no | I did not rerun it. Its θ=0.5 / 0.7 figures (0.785 / 0.504) are well below what replaying the annotator gives (N1). With 67 evaluable falls, about ±0.04 is noise. |

---

## New findings

### N1: The ground truth is produced by the package's own algorithm (`annotate.propose_fall`), and we can replay it
`src/abc2027/annotate.py` ("Proposed fall-interval detector for L2 step A, revision 2") does three things. It builds a 0/1 "state" from the distance to a *standing* profile (the first 1 s of the recording) and a *floor* profile (the last 1.5 s). It finds the 1.5 s window where that state rises most, weighted by motion. It then widens the bounds on frame-difference motion, by at most 3 s to the left and 2 s to the right. The README description ("bounds from the change in static CIR profile on the radar where that change is largest") matches this exactly. The same file also sets `quality` (residual gate) and `needs_review` (start_not_visible, floor_not_visible, event_under_1s, radar_onset_disagree).

Replay results on the aligned 50 Hz data (I wrapped each radar of the aligned array as a `RadarTrack`):

| replay setting | result |
|---|---|
| Full source recording, all 478 falls | \|start error\| median **0.02 s**, 73 % within 0.1 s, 84 % within 0.5 s. `quality` reproduced for **100 %** of falls. `needs_review` reproduced for 87.9 %. |
| **Only the crop visible in the stream**, junctions known, 253 evaluable falls | tIoU ≥ 0.3: **94.5 %**. tIoU ≥ 0.5: **86.6 %**. tIoU ≥ 0.7: **73.1 %**. Median start error 0.02 s. |
| Crop start and end jittered by ±1 s (junction error) | tIoU ≥ 0.5: 85.4 %. ≥ 0.7: 71.9 %. Start 1 s too early: 87.0 % / 73.9 %. **The replay is robust to junction error.** |

The prototype's F1 at θ=0.7 is 0.504. Even this replay with no tuning places 73 % of evaluable falls at tIoU ≥ 0.7, and the remaining gap is mostly the difference between raw timestamps and the aligned data.

**For winning:**
1. Use `propose_fall` (or its state/rise/motion signals) on each detected fall-like segment, as the boundary regressor or as strong features for one. This targets the θ=0.5 term that now decides the ranking.
2. Predict which test falls will be ignore zones (`quality` reproduced 100 %, `needs_review` 88 %). That gives free false-alarm budget: predictions overlapping predicted-flagged falls cost nothing.
3. Hard priors for evaluable falls: **duration ≥ 1.0 s** (0 of the 253 evaluable falls are shorter, while 44 of all 478 are), and onset ≥ 1.0 s after the junction.

Caveat: the replay here assumes the fall segment is already detected. Detection itself is still needed. I scored it with matched-recall proxies, not the official evaluator.

### N2: ADL chunks are bit-exact duplicates; falls and Autre/Empty never are, which identifies ADL and junctions exactly
`ADL_USES = {train: 3, val: 2, test: 2}` means each ADL recording is cut 2–3 times with different random offsets, so the same frames appear in two places in the split. Falls use `fall_uses: 1`, and Autre/Empty segments are inserted once. I hashed pairs of consecutive frames (radar 1) across all 5 val streams, and ignored repeats within 0.5 s of each other, which come from nearest-frame alignment:

| val segments | share of frames repeated elsewhere in the split | segments mostly duplicated |
|---|---|---|
| ADL (463) | **97 %** | 100 % |
| fall (119) | **0 %** | 0 % |
| other (6) | **0 %** | 0 % |

Recovering segments without the composition file, from runs of non-duplicated frames (gaps under 1 s closed, runs over 2 s kept):
- **119 / 119 fall segments** and 6 / 6 Autre/Empty segments are covered, with start error median **0.00 s** (69 % under 0.1 s).
- Only 8 / 463 ADL segments look non-duplicated.
- 52 / 67 evaluable falls lie inside a short (< 25 s) non-duplicated run. The others sit in merged fall→fall or fall→other runs.

**For winning:** on test (ADL_USES = 2, the same as val), this almost certainly removes about 97 % of ADL time as certain negatives and gives exact junction times at every ADL boundary. Combined with N1 it gives a near-oracle pipeline.

**Risk:** this is a generator artifact, not physics. The organizers' `rebuild_test.py` is not in the package, so the test set may have been built differently. **The team lead should decide whether it is within the spirit of the rules.** It only uses the test streams themselves, no external data. In any case, keep a version of the detector that does not depend on it.

### N3: The public seed does not reproduce the published streams
`plan_all(_groups())` with seed 2027 reproduces **0 / 13 train and 0 / 5 val** streams. val_00 matches its first 35 segments and then drifts. The likely cause is that `_groups()` only sees files listed in the composition (the used ADL recordings), so the ADL pool differs from the organizers'.

**For the team:** extra streams generated with this script follow the same statistics but are not exact copies. That is fine for augmentation. Do not assume composition rows can be regenerated in order. When generating extra streams, use ADL_USES = 2 to stay test-like.

The generator also has two fixed rules:
- the 0.2 s crossfade never reaches a fall onset (`protect = event index`);
- it is limited after a fall end (`tail_room`).

As a result, the annotated interval is never blended, as the README states.

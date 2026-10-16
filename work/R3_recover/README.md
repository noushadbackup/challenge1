# R3 `recover`: recovering missed falls without retraining. Result: negative

**Verdict.** No recovery rule passes the promotion rule. The best variants are +0.0003 pooled SR-CV rank F1
(P(Δ>0) = 0.54, 1/4 folds improved), which is far inside TCN seed noise. Most of them change nothing, because the
training-OOF selection usually picks "add nothing".

The base cannot be beaten this way. The missed falls cannot be separated from the many fall-like segments that hold
no fall:
- The segments with high P(fall clip) and no predicted row are mostly ADL. Only about 4 % of them hold an evaluable fall.
- Lower thresholds inside those segments, the arg-max region, the annotator on the segment crop, and an ungated
  decode inside low-gate segments all add more FPs than TPs on the training OOF.

Everything below was run in this package on the frozen `final_norep` bundles. Nothing was retrained.

## Base and reproduction check

The base is `final_norep`, SR-CV TCN bag 0. Its predictions are in `work/STRICT/results/norep`: pooled SR-CV rank F1
**0.9139** (TP/FP/FN @0.3 239/23/14) and official val **0.9373**.

`gather.py` re-runs `uwbfall.predict(..., return_debug=True)` with the bundles in
`/home/user/data_raw/srcv/runs/strict/norep/{fold_0..3,official_train}`. On all 4 folds and on val, the rows are
**byte-identical** to the base CSVs. It also checks two things on each bundle's training streams:
- the subject-grouped OOF segment gate is recomputed with `pipeline._fit_clip` (same code and config);
- gate × raw blend equals the stored gated OOF in `oof.npz` (max abs error < 1e-9).

## (c) Anatomy of the misses

`diag.py` and `diag/events.csv` cover every evaluable fall in the training OOF and in the test streams. A miss is a fall
whose best final row has tIoU < 0.3. The "gate" is P(fall clip) of the junction segment that holds the fall's midpoint.

| set | misses @0.3 | in a segment with **no** row | … of which gate < 0.3 | Chute04 | best tIoU < 0.1 |
|---|---|---|---|---|---|
| SR-CV test (20 streams, 253 falls) | 14 | 6 | 2 | 6 | 11 |
| val (5 streams, 67 falls) | 4 | 2 | 1 | 4 | 3 |
| training OOF, 4 fold bundles | 40 | 15 | 6 | 22 | 24 |
| training OOF, official_train | 10 | 6 | 3 | 5 | 9 |

- **More than half of the misses are not missing candidates.** A row exists in the fall's segment but is mislocated:
  - the detector interval is 1–2 s off and the replay fails to correct it; or
  - the right candidate existed but the 4 s NMS dropped it for a stronger neighbour 2–3 s away (for example
    cv2_test_02 at 1343.8 s).

  No row-adding rule can fix these.
- **The no-row misses come in two kinds:**
  - gate 0.55–1.0 with a gated peak of only 0.20–0.36, below the operating threshold of 0.3–0.4;
  - Chute04 falls whose segment gate is about 0.01–0.03, so the gate cuts their peak to 0.41.
- **High-gate segments with no row hold almost no evaluable falls.** Training OOF, all 5 bundles, gate ≥ 0.5:

  | segment content | segments |
  |---|---|
  | ADL | 178 |
  | pre/post-fall part of a fall clip without the event | 33 |
  | flagged (ignored) fall | 21 |
  | **evaluable fall** | **10** |
  | other clip | 3 |

  If the annotator is run on every such segment, about 190 of the added rows are FPs and 7 are TPs.
  - Gate ≥ 0.95 does not fix it: 29 FP against 2 TP.
  - Only segments whose probability peak is > 0.2 carry signal (7 TP against 9 FP). Rows that land on flagged falls cost
    nothing, which is why some rules look neutral.

### Chute04 (`chute04.py`, training OOF of the 5 bundles)

| type | fall segments | well-segmented (both junctions within 0.6 s) | gate median, well / mis-segmented | share with gate < 0.3, well / mis-segmented |
|---|---|---|---|---|
| Chute01 | 458 | 0.82 | 0.980 / 0.241 | 0.175 / 0.556 |
| Chute02 | 457 | 0.81 | 0.984 / 0.575 | 0.186 / 0.442 |
| Chute03 | 473 | 0.78 | 0.961 / 0.539 | 0.213 / 0.422 |
| **Chute04** | 485 | **0.73** | **0.820** / 0.200 | **0.317** / 0.553 |

Chute04 is hurt twice:
1. It is more often mis-segmented (a missed or extra junction), and then the segment classifier fails for every type.
2. Even when it is well segmented, the segment looks like ADL to the 'phys' segment classifier. Feature medians
   (low-gate Chute04 / high-gate Chute04 / ADL):

   | feature | low-gate Chute04 | high-gate Chute04 | ADL |
   |---|---|---|---|
   | static change across the segment (`static_ab_max`) | −0.59 | −0.25 | −0.80 |
   | motion burst (`diff_max`) | 2.03 | 2.59 | 1.44 |
   | `diff_mean` | 0.59 | 0.88 | 0.36 |

A fall from a chair starts from a low sitting posture. It changes the static radar profile less and has a weaker motion
burst than a standing fall, so its segment features fall between falls and ADL.

The ungated detector does fire on these falls: the median raw blend peak of low-gate Chute04 segments is 0.83. But
1 % of the 5628 low-gate ADL segments also have a raw peak ≥ 0.51, which is why "ungate low-gate segments" fails.

Fixing this needs a retrained segment classifier, with Chute04-aware features or a softer gate for strong detector
peaks, or a better junction detector. That is outside this no-retraining package.

## Rules tested (`rules.py`)

All rules work within one stream. The rules:
- **sec** (idea b): a second decode `to_events(p·[gate ≥ τ], thr2, smooth, min_d2, gap)`. An event is kept only if it
  overlaps no base candidate and is ≥ 4 s from every base row. Grid: τ ∈ {.3,.5,.7,.8,.9,.95}, thr2 ∈ {.1,…,.3},
  min_d2 ∈ {.2,.4,.6,.8}.
- **amax** (idea a, probability version): for a segment with gate ≥ τ, no row and peak ≥ pmin, add one row on the
  region around the arg-max where p ≥ frac·peak. Grid: τ as above, pmin ∈ {.1,…,.3}, frac ∈ {.5,.7}.
- **segcrop** (idea a, literally): the same as amax, but with pmin ∈ {0, .05, .1} and τ up to 0.98, and the row is the
  annotator's output on the segment crop.
- **raw** (the Chute04 path): decode the **ungated** blend inside segments with gate < τ_lo. Grid: τ_lo ∈ {.1,.2,.3,.5},
  thr_r ∈ {.3,…,.7}, min_d2 ∈ {.4,.8}.

Every rule also chooses its boundary mode on the training data:
- `det`: the detector interval;
- `rep`: the base replay stage on the interval (6 crops on distinct frames, gate 0.3);
- `seg`: `abc2027.annotate.propose_fall` on the whole junction segment, on distinct frames, as in the base replay.

`replay.py` computes these boundaries for every interval any grid can propose. Added rows get score 0.1·peak, which is
below every base row, so they only match falls that are otherwise unmatched. The base 4 s NMS then runs on the union.

**Selection rule, fixed before any test scoring (`rules.select`):**
1. For each bundle, choose the config with the largest delta on **that bundle's own training OOF**. The OOF has
   stream-grouped TCN/HGB probabilities, a subject-grouped segment gate, and the in-process official evaluator with
   `--ignore-flagged`.
2. Ties go to the config with the fewest added rows.
3. If no config improves the training OOF, the bundle stays the base.

The val prediction uses the `official_train` bundle's own choice. No test stream and no val label was used for any
choice. The test-side counts in §(c) are descriptive only.

## Results

Training OOF: the best delta of any config per bundle is +0.000 to +0.009 rank F1 (0–2 TP). Averaged over the grid,
every rule and every boundary mode is ≤ 0 (for example sec/det −0.0027, amax/rep −0.0004, raw/seg −0.0018).

SR-CV (pooled over 4 subject-disjoint folds, 253 falls) and official val, against the base, with `srcv.promote`
(5000-draw paired subject-cluster bootstrap):

| variant | pooled rank F1 | TP/FP/FN @0.3 | Δ [95 % CI] | P(Δ>0) | Δ per fold | folds improved | val | rows added (folds 0–3, val) | promote |
|---|---|---|---|---|---|---|---|---|---|
| base `final_norep` | 0.9139 | 239/23/14 | – | – | – | – | 0.9373 | – | – |
| sec | 0.9142 | 240/24/13 | +0.0003 [−0.006, +0.006] | 0.54 | 0 / 0 / +.0015 / 0 | 1 | 0.9383 | 0,0,3,3, 11 | no |
| amax | 0.9142 | 240/24/13 | +0.0003 [−0.006, +0.006] | 0.54 | 0 / 0 / +.0015 / 0 | 1 | 0.9383 | 0,0,3,3, 9 | no |
| amax_rep (mode fixed to rep) | 0.9142 | 240/24/13 | +0.0003 [−0.006, +0.006] | 0.54 | 0 / 0 / +.0087 / −.0074 | 1 | 0.9383 | 0,0,3,3, 9 | no |
| amax_seg (mode fixed to seg) | 0.9142 | 240/24/13 | +0.0003 [−0.006, +0.006] | 0.54 | 0 / 0 / +.0015 / 0 | 1 | 0.9383 | 0,0,3,3, 8 | no |
| segcrop | 0.9142 | 240/24/13 | +0.0003 [−0.006, +0.006] | 0.54 | 0 / 0 / +.0015 / 0 | 1 | 0.9373 | 0,0,3,2, 0 | no |
| sec_rep (mode fixed to rep) | 0.9121 | 239/24/14 | −0.0018 [−0.007, 0] | 0.00 | 0 / 0 / 0 / −.0074 | 0 | 0.9383 | 0,0,0,3, 11 | no |
| raw | 0.9139 | 239/23/14 | 0 | 0 | 0 / 0 / 0 / 0 | 0 | 0.9453 | 0,0,0,0, 2 | no |

- On SR-CV, every variant recovers at most one fall: fold 3, +1 TP and +1 FP @0.3.
- The val gain of `raw` (+0.008, two added rows) is a single bundle's choice. On SR-CV the same rule never fires,
  because no fold bundle's training OOF supports it. It is not evidence.

Also checked on the training OOF and dropped before any test run: putting back the candidates that the 4 s NMS removed,
at low score. Over the 5 bundles that adds 2 missed falls against 21 FPs (+4 on flagged falls or duplicates).

## Reproduce

Every python call needs this environment:

```bash
cd /home/user/challenge1
export PYTHONPATH=/home/user/challenge1/work/R3_recover:/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src \
       ABC_ROOT=/home/user/data_raw/abc2027_uwb_train_v1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
```

The steps:
1. `python3 work/R3_recover/gather.py` dumps p, gate and segments for train OOF and test. It takes about 12 min and
   checks byte-identity with the base.
2. `python3 work/R3_recover/diag.py` prints the miss anatomy and writes `diag/events.csv` and `diag/segments.csv.gz`.
3. `python3 work/R3_recover/chute04.py` runs the Chute04 segment analysis and writes `diag/chute04_segments.csv.gz`.
4. `python3 work/R3_recover/replay.py` computes replay and segment-crop annotator bounds for all proposable intervals.
   It takes about 6 min.
5. `python3 work/R3_recover/run.py [variant …]` tunes on training OOF, predicts, scores and promotes. It writes
   `results/<variant>/{fold_k,val}.csv` and `summary.json`.

Large intermediates (31 MB of pickles and the tuning tables `tab_*.csv`) are in `/home/user/data_raw/r3/recover/`.

## Compliance and notes

- Everything runs on one stream at a time: its own probabilities, junction segments and raw CIR crops. Nothing is
  compared across streams or with train/val content.
- No repeat-statistic features are used. The annotator runs on distinct frames exactly as the base replay does
  (`final_norep` keeps the replay dedup), so this adds no new grey-zone element.
- `uwbfall.predict` inside `gather.py` runs with the bundle's own `runtime.threads = 2`. That is needed for byte-identical
  TCN outputs, and it ran for about 1 min per target. Everything else ran on 1 thread under `nice`.
- Nothing outside `work/R3_recover/` and `/home/user/data_raw/r3/recover/` was written. Nothing was committed.

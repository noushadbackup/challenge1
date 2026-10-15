# Rule compliance after the organisers' answers (Q1, Q2)

## What the organisers said

* **Q1:** the test streams use the same generator and settings as train/val: 3–8 s / 5–10 s margins, 0.2 s
  crossfade, ~58 falls/h, streams up to 30 min. Only the seed and the participants differ.
  → SR-CV mirrors the test; the expected test level stands.
* **Q2:** identifying repeated content is **not allowed**. That means matching, comparing or aligning content across
  test streams, within a test stream, or with train/val segments, "in order to detect repeated portions". Each test
  stream must be processed independently, and falls must be detected from the signal itself.

## What our system does

| element | in FINAL | assessment |
|---|---|---|
| cross-stream ADL duplicate mask | **off** (pinned in config) | compliant |
| matching against train/val segments | never | compliant |
| each test stream processed independently | yes (predict is per stream) | compliant |
| segment classifier features `rep_rate`, `rep_sp2/3/4`, `rep_n`: share and spacing of bit-exact *consecutive* identical frames (50 Hz sample-and-hold) inside one stream | **on** | grey zone: compares frames within a stream; acts as a recording-session fingerprint |
| boundary replay on distinct frames (`dedup`): drops a frame identical to the previous one before running the annotator | **on** | grey zone: compares consecutive frames within a stream, though only to restore the true frame sequence |
| junction detector, segment length, static-profile change | on | signal-based; not about repeated content (junctions are described by the organisers themselves) |

## Measured cost of removing the grey-zone elements

SR-CV (4 subject-disjoint folds, 253 evaluable falls, official evaluator, bag 0) and official val, from re-tuning the
FINAL fold bundles (HGB/TCN unchanged). Script: `work/STRICT/run_strict.py`; outputs: `work/STRICT/results/`.

| variant | what is removed | SR-CV | Δ vs FINAL [95% CI] | P(gain) | folds improved | val |
|---|---|---|---|---|---|---|
| **final** | – | 0.9183 | – | – | – | 0.9524 |
| **final_norep** | repeat-statistic features | 0.9139 | −0.004 [−0.013, +0.003] | 0.13 | 0/4 (2 unchanged) | 0.9373 |
| final_nodedup | replay dedup | 0.9001 | −0.018 [−0.030, −0.007] | 0.001 | 0/4 | 0.9524 |
| **final_strict** | both | 0.8958 | −0.023 [−0.035, −0.010] | 0.000 | 0/4 | 0.9373 |

* Dropping the repeat-statistic features costs very little (within noise).
* Dropping the dedup costs a clear ~0.018.

## Recommendation

1. **Submit `final_norep` by default.** It loses about 0.004 and removes the features closest to the wording of Q2.
2. **Ask the organisers about the dedup (one precise question, draft below).**
   * If they allow it, stay on `final_norep`.
   * If they forbid it, switch to `final_strict` (expected about −0.02).
3. Keep `final` only if the organisers explicitly allow sample-and-hold frame statistics as features.

Commands (every variant ships as a verified bundle in `models/<variant>/`, with sha256):

```bash
make predict-test VARIANT=final_norep TEST_DIR=/path/to/streams/test          # default recommendation
make predict-test VARIANT=final_strict TEST_DIR=/path/to/streams/test         # if dedup is not allowed
bash run_all.sh cuda --variant final_norep --data /mnt/e/uwbfall/data_raw     # full local rebuild of a variant
```

All three variants passed `verify-models`, `dry-run` (`SUBMISSION CHECK: OK`, 0 drift alerts) and `checks`.

## Draft follow-up email (abc2027@autocare.ai, subject "ABC Challenge")

> Dear organisers,
>
> Thank you for the clear answers. One follow-up on Q2, to make sure our method is compliant.
>
> The published streams contain bit-exact consecutive duplicate frames: about 25 % of frames on each radar repeat the
> previous frame, which we understand comes from aligning each radar to the common 50 Hz grid. Our pipeline processes
> each test stream on its own and never compares content across streams or with the training/validation segments.
> Within a stream, we would like to:
>
> (a) drop a frame when it is identical to the immediately preceding frame of the same radar, before computing motion
>     features, so as to work on the radar's true frame sequence (about 37 Hz); and
> (b) use the rate of such consecutive duplicate frames as a signal-quality feature.
>
> Neither step searches for repeated portions of activity recordings; each only compares a frame with the one right
> before it. Are (a) and/or (b) allowed?
>
> Best regards,
> [team name]

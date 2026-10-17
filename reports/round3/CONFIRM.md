# Confirmation of the round-3 candidates on fresh TCN seeds

Run on the team's PC with `make confirm DEVICE=cuda` (`tools/confirm.py`, 2026-10-17).
- Seed sets 3, 4 and 5 (seed_offset 30000/40000/50000) were never used for any selection.
- SR-CV: 253 evaluable falls, official evaluator, paired subject-cluster bootstrap against the base on the same seeds.

| Variant | SR-CV rank F1 | Delta [95% CI] | P(gain) | Folds improved | Val | F1@0.5 | FP@0.3 | Median start err (s) | Passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_norep (submitted) | 0.9150 | - | - | - | 0.9434 | 0.884 | 20.7 | 0.04 | - |
| no range bands | 0.9039 | -0.0112 [-0.0403, +0.0100] | 0.234 | 2/4 | 0.9493 | 0.867 | 28.3 | 0.04 | no |
| boundary blend | 0.9212 | +0.0062 [-0.0000, +0.0126] | 0.974 | 2/4 | 0.9534 | 0.888 | 18.3 | 0.16 | no |
| no range bands + boundary blend | 0.9075 | -0.0076 [-0.0382, +0.0159] | 0.330 | 2/4 | 0.9559 | 0.869 | 26.7 | 0.15 | no |

## Decision: no change; `final_norep` stays the submission

- **No range bands.** The ablation's +0.009 (seed sets 0–2) reverses to −0.011 on fresh seeds. It was seed noise
  picked out of 8 ablations.
- **Boundary blend.** The sign is consistent: +0.0057 on seed sets 0–2 and +0.0062 on seed sets 3–5. On the fresh seeds
  it fails the fold clause (2/4 folds), and it raises the median start error (tie-break 1) from 0.04 s to about
  0.16 s. The gain is about one event out of 253.
  - The pre-registered rule says no, so it stays an optional, documented setting (`boundary.det_blend`), off by default.
  - It is worth mentioning in the paper as a consistent but small effect.

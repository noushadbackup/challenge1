# Ablation of the submitted system (`final_norep`) on SR-CV

Run on the team's PC (RTX 5060 Ti, `make ablation DEVICE=cuda BAGS=3`, 2026-10-16) with `tools/ablation.py`.
Each variant is `configs/final_norep.json` with one component removed. It is refitted on every SR-CV fold, or re-tuned
for post-model stages, with TCN seed sets 0, 1 and 2 (seed_offset 0 / 10000 / 20000).
- Scoring: official evaluator, pooled over 4 subject-disjoint folds (253 evaluable falls).
- Delta: paired subject-cluster bootstrap with 5000 draws, ablated minus base.
- GPU training is not bit-identical to the CPU bundles. The 3-bag base, 0.908, is therefore a little below the CPU bag-0
  value of 0.914 (TCN seed sd 0.004–0.013).

## 3 TCN seed sets (paper table)

| Variant | SR-CV rank F1 [95% CI] | Delta [95% CI] | P(component helps) | Folds where removal does not help | F1@0.5 | FP@0.3 | Median start err (s) | Val |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| submitted system (final_norep) | 0.908 [0.864, 0.942] | - | - | - | 0.872 | 22.7 | 0.04 | 0.946 |
| - background fingerprint | 0.919 [0.889, 0.945] | +0.011 [-0.010, +0.041] | 0.22 | 2/4 | 0.883 | 14.0 | 0.04 | 0.932 |
| - structure and fingerprint features | 0.914 [0.883, 0.942] | +0.006 [-0.014, +0.031] | 0.30 | 2/4 | 0.880 | 18.7 | 0.04 | 0.924 |
| - range-band channels | 0.917 [0.875, 0.949] | +0.009 [+0.001, +0.017] | 0.01 | 1/4 | 0.880 | 21.7 | 0.04 | 0.934 |
| - TCN (HGB only) | 0.847 [0.762, 0.905] | -0.060 [-0.143, -0.006] | 0.99 | 3/4 | 0.822 | 17.0 | 0.04 | 0.922 |
| - HGB (TCN only) | 0.901 [0.852, 0.940] | -0.006 [-0.018, +0.004] | 0.88 | 2/4 | 0.863 | 25.7 | 0.04 | 0.946 |
| - segment gate | 0.902 [0.846, 0.943] | -0.006 [-0.022, +0.007] | 0.79 | 3/4 | 0.872 | 28.0 | 0.04 | 0.950 |
| - 4 s time NMS | 0.895 [0.848, 0.932] | -0.013 [-0.021, -0.007] | 1.00 | 4/4 | 0.858 | 31.3 | 0.04 | 0.941 |
| - annotator replay | 0.894 [0.849, 0.929] | -0.014 [-0.030, +0.000] | 0.97 | 3/4 | 0.820 | 21.3 | 0.32 | 0.931 |

## Reading

- **Clearly useful:**
  - **TCN.** −0.060 without it; it carries detection.
  - **4 s NMS.** −0.013, 4/4 folds; it removes duplicate rows.
  - **Annotator replay.** −0.014; F1@0.5 drops from 0.872 to 0.820, and the start error rises from 0.04 s to 0.32 s,
    which is the first tie-break.
- **Probably useful, not significant:** the HGB member (−0.006, P 0.88) and the segment gate (−0.006, P 0.79).
- **Not useful on SR-CV:**
  - **Range-band channels.** Removing them gives +0.009 [+0.001, +0.017], P(gain) 0.99, and 3 of 4 folds improve.
    Val drops 0.012, which is within the 0.02 tolerance. This meets the promotion rule as a change. It was
    found by looking at 8 ablations, so it is re-tested on fresh seeds before any decision (`tools/confirm.py`).
  - **Background fingerprint and structure features.** Removing them gives +0.011 / +0.006, but the CIs include 0
    and only 2 of 4 folds improve, so these are not significant. Without the fingerprint, FPs fall from 22.7 to 14.0
    and val falls from 0.946 to 0.932.
- 1 seed set alone gives the same signs but noisier numbers, e.g. fingerprint +0.009 and bands +0.003. This is
  why 3 seed sets are reported.

# V3: Data scientist (spatial and radar geometry, physics of falls). Verification report

Role: re-verify F2, F5 and F6 independently, vote on F1 to F10, and look for new geometry and physics patterns.
All scratch code is in `/home/user/data_raw/agents/V3/`. Every number below comes from those scripts in this session.

## What I ran (different method from the lead's where possible)

| script | what it does |
|---|---|
| `f2_timing.py` | Matches each event to the fall segment that contains it, directly from `streams_events.csv` and `streams_composition.csv`. Computes pre and post margins, segment lengths, overlaps and falls per hour, for all falls and for evaluable falls only. |
| `f56.py` + `an.py` | Builds a complex profile from `prof_02` (I + jQ, 0.2 s blocks). **Motion = block-to-block complex difference \|Δz\|² per bin.** This is not the EMA residual or `mot_map` the lead used. It is normalised per radar by the median of the stream's ADL blocks, excluding blocks within 1 s of a junction. Also computes the per-event dominant radar and bin, and the motion-peak time with two motion measures (smoothed frame-difference `e_diff`; EMA `e_mot`). **Static change = normalised complex distance between mean profiles in [t_s−1.5, t_s−0.5] and [t_e+0.5, t_e+1.5].** Negatives: ADL pairs inside one chunk, Autre/Empty pairs, **hard negatives inside the same fall clip (both windows before the fall, or both after it)**, and pairs that straddle a junction. |
| `new.py` + `an2.py` | Localises high-motion blocks to a (radar, range bin) cell, and compares a physics-derived fall interval with the annotation. |

## Votes

| F | vote | conf | by code | evidence |
|---|---|---|---|---|
| F1 I/Q split | agree | 0.85 | no | Not tested by me. V1 and V2 checked it at sample level. My complex features gave clean, localised results (below), which is consistent with this split. |
| F2 timing | **agree** | 0.95 | **yes** | All 478 events sit inside exactly one fall segment (478 fall segments). Pre-margin: median 3.03 s, range 0.00–7.86 s. Post-margin to segment end: median 7.56 s, range 3.26–10.00 s (7.36 s to the next segment start, because of the 0.2 s overlap). Fall segment length 7.32–18.64 s (median 12.40 s). Duration median 1.74 s, range 0.48–5.0 s. Autre/Empty 77–120 s (median 118 s, n = 24). 58.1 falls/h over 8.23 h. Overlaps: 2302 at 0.20 s, 24 at 0 (hard cuts). **Nuance:** for evaluable falls the pre-margin is ≥ 1.00 s (median 3.20 s) and the post-margin is uniform on 5.0–10.0 s. This matches `MARGIN_AFTER = (5, 10)` in `streams.py`. Only **30.7 evaluable falls/h**. |
| F3 EMA junction spike | agree | 0.8 | partial | I did not recompute the 84× figure. The related static jump at junctions is huge: median complex profile change 1.09, against 0.115 for ADL pairs and 0.78/0.89 for D1/D2 falls. |
| F4 junction detection AUC 0.97 | agree | 0.7 | no | This is plausible given the junction static-change size above. V2 reports an even stronger detector. |
| F5 mat to radar | **partially** (stronger than claimed for D1/D2; wrong for D3) | 0.9 | **yes** | Per event, the dominant radar by block complex-difference motion is: **D1 → R29444 for 155/160 (97 %), D2 → R30722 for 154/159 (97 %), D3 → R29171 for 158/159 (99 %)**. It is stable across train and val (val: 40/40, 40/40, 38/39). Median motion relative to ADL: D1 307× on R29444; D2 518× on R30722; **D3 16.7× on R29171 but only 0.2–0.3× on the other two radars**. Peak bin on the dominant radar: D1 13 (IQR 9–21), D2 11 (IQR 9–14), so roughly 0.5–0.7 m, which agrees with "bins 10–13". **D3 is not "weak on all radars": it is consistently seen by R29171 at bin ~43 (IQR 41–46, ≈2.2 m).** |
| F6 fall vs motion peak; static change | **partially** | 0.85 | **yes** | (a) Static-change AUC against ADL pairs, by my own method: D1 0.983 (train) / 0.986 (val); D2 0.992 / 0.968; D3 0.64 / 0.63 (the lead reports 0.76). This part is confirmed for D1/D2. (b) **The claim "the annotation is not at the motion peak, which is ~1.5 s before onset" is an artefact of the EMA motion measure.** With the EMA residual over the whole segment I reproduce it (peak inside the fall: 10 % for D1, 42 % for D2; the lead has 11 % / 35 %). With junction-immune smoothed frame-difference energy, excluding 0.6 s at each segment edge, **the peak lies inside the annotated fall for 92 % (D1), 89 % (D2) and 89 % (D3) of falls, with a median of +0.73 s (D1) and +1.10 s (D2) after onset.** The lead's last sentence in F6 agrees with this. I could not find the code that produced the 11 %/35 % number in `analysis/`. (c) Static change is **not specific** (see N2). |
| F7 corpus fingerprint | agree | 0.7 | no | Not tested. It is consistent with my N1: ADL motion occurs on a different radar mix than fall-room motion. |
| F8 D3 flagged | agree | 0.95 | yes | From my event table: D3 train 16 of 120 evaluable (87 % flagged); val 0 of 39; val evaluable D1 33 + D2 34 = 67. |
| F9 activity before a fall is random | agree | 0.75 | no | This is plausible from the shuffled generator. Not re-tested. |
| F10 prototype 0.884 | partially | 0.6 | no | I read `prototype_detector.py`. Labels come only from `frame_labels`, and the structure features come from *detected* jumps, not from the composition CSV, so I see no obvious leakage. However, 67 falls give roughly ±0.04 noise. The PCA fingerprint is also fit on whatever frames are passed in, so check that val is not included in the fit. |

## New findings

### N1: Each mat is one (radar, range-bin) cell, but the fall-room hard negatives share that cell
High-motion 0.2 s blocks (>100× ADL median on the best radar, >1 s from a junction):

| block type | n | R29171 / R29444 / R30722 share | in mat cell |
|---|---|---|---|
| fall D1 | 891 | 0.007 / **0.982** / 0.011 | 0.73 |
| fall D2 | 1337 | 0.004 / 0.000 / **0.996** | 0.91 |
| fall D3 | 132 | **1.000** / 0 / 0 | 0.46 |
| ADL | 5989 | 0.35 / 0.40 / 0.25 | 0.39 |
| fall clip, outside the fall | 3025 | 0.04 / 0.29 / 0.67 | **0.81** |
| Autre | 2494 | 0.18 / 0.17 / 0.66 | 0.58 |

Mat cells used: R29444 or R30722 bins 5–24, R29171 bins 35–51.

- The radar and range bin pin down *where* motion happens. During a D1/D2 fall that place is almost deterministic: 98–99.6 % of the blocks are on the mat's radar.
- This separates falls from ADL motion well.
- It does **not** separate a fall from walking near the mat in the same recording. Non-fall motion inside fall clips is 81 % in-cell.

Implication: use `(dominant radar, peak bin)` as a cheap gate or feature against ADL. To reject hard negatives, the model needs *temporal* shape: burst → lying still, plus the static change. Model D3 on R29171 bins ~40–46, not "everywhere".

### N2: Static change is non-specific. Junctions beat falls, and D2 post-fall drift is a weak spot
- **Junction-straddling pairs have a larger static change than falls**: fall vs junction AUC is 0.32 (D1), 0.38–0.39 (D2) and 0.08–0.13 (D3). 74 % of junctions exceed the median D1/D2 fall change.
- Hard negatives inside the same fall clip:
  - D1: fall vs before-fall pairs 0.98; fall vs after-fall pairs 0.97–0.98.
  - **D2 val: 0.72 vs before-fall (n = 4), 0.75 vs after-fall (n = 32).** D2 train: 0.93 / 0.90.
  - Autre/Empty: 0.92–0.98.

Implication: the static-change feature must be computed *within* a detected segment, never across a junction (V2's N2 junction detector is a prerequisite). Expect D2 boundary and false-alarm errors from post-fall settling or drift. Note also that the **unflagged D3 falls are clearly visible**: static-change AUC against ADL is 0.974 for the 16 evaluable train D3 falls, against 0.64 for all D3. Flagged D3 falls look like the "invisible" ones, so a test D3 fall that does count will probably be detectable.

### N3: The annotated start is physical; the annotated end is not
I computed the motion in the mat cell on the mat's radar (0.6 s smoothing), found its peak inside the segment, and expanded left and right to 10 % of the peak:

| | start error median (MAD) | end error median (MAD) | IoU ≥ 0.5 |
|---|---|---|---|
| D1 (n = 122) | **+0.01 s (0.17)** | +0.72 s (0.39) | 0.62 |
| D2 (n = 115) | +0.02 s (0.64) | +1.06 s (0.64) | 0.38 |

- The physical burst duration correlates with the annotated duration at only **r = 0.08 (D1) and 0.11 (D2)**.
- Motion continues 0.7–1.1 s after the annotated `t_end`.
- The motion peak lies at 59 % (D1) / 72 % (D2) of the annotated duration.

Implication for θ = 0.5/0.7 and the tie-break: regress `t_start` from the burst onset, since D1 has a 0.17 s MAD, which is good for the median start-error tie-break. Predict `t_end` mostly from a learned duration prior plus offset rather than from motion settling. Together with V1's 2.00 s duration spike, this means the end is a convention, not a physical event. A naive "start at the 10 % crossing, end at the 10 % crossing" would lose about 0.7–1 s at the end and push IoU below 0.5 for D2.

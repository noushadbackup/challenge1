# V6 — Exploratory data scientist (per fall type / per subject / per radar)

**Role:** independent re-verification of F1, F5 and F7, plus a search for new patterns by fall type, subject and radar.
All numbers below come from scripts I ran in this session. The scripts are in `/home/user/data_raw/agents/V6/`.

| script | what it does |
|---|---|
| `f1_iq.py`, `f1_offset.py` | Reads raw h5 slices (3 s around 20–25 val D1/D2 falls on the mat's radar). Tests whether channel *k* pairs with *k+d* like a complex I/Q pair, using the improperness coefficient \|E[z²]\|/E[\|z\|²] for every offset d = 1..367 |
| `f5_radar.py` | Per fall: complex 0.2 s block-difference energy (from `prof_02`, junction blocks removed) per radar and range bin, normalised by that radar's ADL median in the same stream. Output: `f5_per_fall.csv` |
| `f7_fingerprint.py` | Standardised logistic regression (not GBM) on log 1 s static amplitude. Train on train, AUC on val. Variants: full, shape-only, scale-only, single radar. Junctions (±1.5 s) and falls (±1.5 s) are excluded |
| `proto_breakdown.py` | Breaks down the lead's saved val predictions (`reports/prototype/val_full.csv`) by fall type and subject |
| `fall_type_profile.py` | Onset-aligned motion profile per fall type (D1/D2, train+val) |
| official evaluator on `val_full.csv` | Re-scored the lead's CSV: **ranking F1 0.8840**, @0.3 TP 62 / FP 6 / FN 5 (matches F10) |

## Votes

| F | vote | conf | by code | evidence |
|---|---|---|---|---|
| F1 | agree | 0.95 | yes | I scanned all partner offsets d. The improperness coefficient has a clear minimum at **d = 184 (0.335)**; neighbours 183/185 give 0.42/0.40, every other offset has median 0.995 (real-like), and d = 1 gives 0.91. An interleaved pairing (2k, 2k+1) gives r = 0.92 and an eigen-ratio of 0.04, so it is collinear, not quadrature. var(Q_k)/var(I_k) has median 1.09. Phase velocity is coherent across adjacent bins (r = 0.62, vs 0.47 with Q shuffled in time). Conclusion: I = 0..183, Q = 184..367. |
| F2 | agree | 0.75 | no | Not re-derived. It is consistent with the composition sums I computed (fall segments about 12.1–12.8 s on average per type; Autre/Empty 875–936 s per 8 segments, i.e. about 110–117 s each). |
| F3 | agree | 0.65 | no | Plausible from the EMA definition (`extract_features.py` / baseline); not re-measured. |
| F4 | agree | 0.6 | no | Not re-measured. AUC 0.97 at ±1.5 s looks plausible, but recall 0.82 means about 1 in 5 junctions is missed, so any segment logic must be soft. |
| F5 | **partially** | 0.9 | yes | D1 → R29444 for **98%** of falls (median 278× ADL), D2 → R30722 for **100%** (510×). Peak bins: D1 median 9 (IQR 5–15.5), D2 median 11 (IQR 8–18). This holds for every one of the 16 D1/D2 subjects (hit 0.95–1.00). **But D3 is not "weak on all radars"**: **100% of D3 falls are seen best by R29171 (index 0)** at **far bins 45–61 (≈2.3–3.1 m)**, median 14× ADL (by subject: 5–29×, and **Test_P3_3 153×**). On R29444/R30722, D3 is *below* ADL level (0.1–0.4×). |
| F6 | agree | 0.7 | partly | Fall types 1–3 have their block-difference energy peak 0.8–1.1 s after onset, which matches "0–2 s after onset". The static-change AUCs were not re-run. **Chute04 is an exception** (see N2). |
| F7 | **partially** | 0.8 | yes | The corpus fingerprint is real and *stronger* with a linear model. Fall-recording margins vs ADL: val AUC **0.980** (per val stream 0.92–1.00; D1 1.00, D2 1.00, D3 0.93). It survives shape-only normalisation (0.977) and is weak on scale-only (0.71), so it is room/geometry, not gain. **"Fall vs Autre/Empty ≈ chance (0.56)" did not reproduce**: full model 0.74, and **R29444 alone gives 0.86**. Caveat: val has only about 6 Autre/Empty segments, so this AUC is noisy. Autre/Empty vs ADL is 0.99. |
| F8 | partially | 0.8 | yes (flag rates) | The counts are right (the 16 evaluable train D3 falls all come from **Test_P3_3**; val D3 is 100% flagged). But flagging depends on the subject: **7 of 8 D3 subjects are 100% flagged, while P3_3 is only 20% flagged**. A test D3 subject like P3_3 would make D3 count. The README's flag rule is a residual-energy gate, and P3_3 is the strongest D3 subject on R29171 (153×). Do not drop D3 from the design. Keep a radar-0, far-bin pathway. |
| F9 | agree | 0.7 | no | Matches the generator (random shuffle); not re-tested. |
| F10 | agree | 0.85 | yes (re-scored) | The official evaluator on the saved `val_full.csv` gives 0.8840, TP/FP/FN 62/6/5 @0.3. Leakage was not audited line by line. ±0.04 noise remains (67 falls). |

## New findings

### N1 — D3 lives on radar R29171 at 2.3–3.1 m; unflagged D3 depends on the subject
- Data: `f5_radar.py`. D3 falls: best radar = index 0 for 100% of them. Peak bin median 60 (IQR 45–61). Median ratio 14× ADL; P3_3 153×, P3_2 28.5×, the rest 5–18×.
- Flag rate per D3 subject: 1.00 for P3_1/2/5/6/7/8/9 and **0.20 for P3_3**.
- On the train OOF CSV the prototype already gets 15 of 16 P3_3 falls (IoU ≥ 0.3). This is unverifiable on val (no evaluable D3).
- **Why it matters:** if one test subject behaves like P3_3, up to about a third of test falls could be evaluable D3. Features must include radar 0 at bins 40–65. Monitor D3 with leave-P3_3-out CV, since val cannot measure it.

### N2 — Chute04 (chair fall) is the boundary bottleneck
- Prototype on val (`proto_breakdown.py`, best-IoU prediction per evaluable fall):

| type | n | hit@0.3 | hit@0.5 | hit@0.7 | median IoU | median \|end err\| |
|---|---|---|---|---|---|---|
| Chute01 | 14 | 0.93 | 0.86 | 0.57 | 0.75 | 0.36 s |
| Chute02 | 19 | 1.00 | 1.00 | 0.58 | 0.73 | 0.30 s |
| Chute03 | 20 | 1.00 | 0.80 | 0.70 | 0.76 | 0.17 s |
| **Chute04** | 14 | **0.71** | **0.43** | **0.07** | **0.47** | **0.70 s** |

  Chute04 accounts for 4 of the 5 misses @0.3 and 8 of the 14 misses @0.5. The train OOF CSV (`tmp_train.csv`, last tuning run) shows the same: hit@0.5 0.36 vs 0.63–0.69.
- Physics (`fall_type_profile.py`, D1/D2, train+val): for Chute04 the motion peak is at onset (+0.11 s), vs +0.8–1.1 s for the other types. Only **36%** of the motion energy falls inside the GT interval (vs 67–70%). The active-motion end has IQR **3.05 s** (vs 0.8–1.1). Chute04 also has the weakest motion: 102× (D1) / 220× (D2) vs 310–650× for the other types. GT duration is the longest (median 2.0 s).
- **Why it matters:** θ = 0.5/0.7 is in the ranking. A type-agnostic boundary regressor is pulled toward types 1–3 and misplaces chair falls. Options: (a) add an auxiliary 4-class fall-type head (labels are available in train) and use type-conditioned start/end offsets; (b) weight Chute04 more in the boundary loss; (c) report per-type hit@0.5 in every CV run.
- Predicted durations barely follow GT: corr(GT duration, predicted duration) = 0.31, and predictions cluster at 1.6–2.8 s. Duration regression is weak overall.

### N3 — Per-subject variation is large, but the radar → mat mapping is stable
- Motion ratio on the expected radar ranges from 122× (P1_3) to 1198× (P2_8) by subject. The separation from the second-best radar ranges from 16× to 298×. Val subjects P1_8/P1_9 are the *easiest* (sep 248–299×).
- Val is therefore probably optimistic for weaker test subjects. Prefer per-stream or per-radar robust normalisation (ratio to the stream's ADL-like median) over absolute energy thresholds.
- Prototype hit@0.5 by val subject: 0.67 (P1_9) to 0.89 (P2_8). Subject variance is about the same size as the ±0.04 noise.

## Caveats
- The F7 Autre/Empty AUC rests on only about 6 val segments.
- `tmp_train.csv` is whatever the last OOF tuning iteration wrote, not necessarily the tuned OOF.
- The "test subject = index 4" idea (index 4 is missing from train and val subjects) is a guess; the README does not say so.

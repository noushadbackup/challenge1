# V2 — Signal-processing data scientist (radar / IQ): verification report

**Role:** independent re-verification of F1 (I/Q layout), F3 (EMA junction artefact), F4 (junction detectability), votes on F1–F10, and new signal-level patterns.
**Rule followed:** every number below comes from scripts I wrote and ran in this session (`/home/user/data_raw/agents/V2/*.py`, outputs `*.json` next to them). I did not re-run the lead's scripts. Package and REPORT/PLAN untouched.

## What I ran

| script | method (different from lead's where possible) | output |
|---|---|---|
| `f1_iq_check.py`, `f1_offset_scan.py` | Raw h5 (val_00, train_03). **Doppler sideband asymmetry** of z = x[k] + j·x[k+184+o] in the 24 strongest-motion 4 s windows (a real signal has a symmetric spectrum, a true I/Q pair does not), **non-circularity** \|E z²\|/E\|z\|², range roughness of the envelope, far-bin noise stats. Compared against interleaved (2k,2k+1), within-I (k,k+1), reversed and offset pairings o = −4..4. | `f1.json`, `f1_offset.json` |
| `f3_junction_ema.py` | Rebuilt the official energy baseline from cached `e_mot` (p98, merge 0.8 s, min 0.6 s). **Official evaluator gives 0.3151, TP 34 / FP 100 / FN 33 @0.3**, the same as README, so the rebuild is exact. Matched with the evaluator's own `_match_stream` and measured each FA's distance to the nearest junction against a random-time null. Per-junction peak = max EMA energy in the 1 s after the junction ÷ median EMA energy of ADL frames > 5 s from any junction. | `f3.json`, `energy_baseline_val.csv`, `energy_eval.json` |
| `f4_junction_detect.py` | Two-sided change-point on the 0.2 s **complex** profile (`prof_02`): ‖mean(1 s after) − mean(1 s before)‖², either plain or **normalised by the variance inside each window** (Welch-t style). Threshold chosen on train only, peaks after ±2.5 s NMS, matched to junctions at ±1.5 s and ±0.5 s. | `f4.json` |
| `n1_duplicate_frames.py`, `n1b_freezes.py` | Exact repeated frames (`e_diff == 0`) from the package's nearest-frame 50 Hz alignment: their rate, the spacing between repeats, freezes, and how all three differ by corpus. | `n1.json`, `n1b.json` |
| inline check | F2 / F8 numbers recomputed from the CSVs | (stdout, numbers quoted below) |

## Votes

| # | vote | conf. | by code | evidence |
|---|---|---|---|---|
| F1 I/Q = halves | **agree** | 0.95 | yes | Sideband asymmetry: halves **0.27** (val_00) / **0.29** (train_03); interleaved 0.07, within-I 0.069, independent-noise null 0.078. In the offset scan both asymmetry and properness peak **exactly at o = 0** (non-circularity is lowest at o = 0: 0.296 / 0.31, against 0.65–0.68 at o = ±4). Far-bin noise: I and Q have equal std (e.g. 0.043 vs 0.040 ×1e-3), corr(I_k,Q_k) ≈ 0 (−0.03…0.02), while corr(I_k,I_k+1) = 0.73–0.82, as expected for quadrature channels. A single I half changes sign 31× over 184 bins (carrier), and the halves envelope is smoother than \|I\| (roughness 0.167 vs 0.221). |
| F2 timing | agree | 0.9 | yes (CSV) | 478 falls: pre-margin 0–7.86 s (median 3.03), post 3.26–10.0 s (median 7.56), fall segment 7.32–18.64 s, Autre/Empty 77–120 s, median duration 1.74 s, **58.1 falls/h**. |
| F3 EMA junction spike, 90 % FA near junction | **agree (magnitudes are definition-dependent)** | 0.9 | yes | **90 % of the 100 val FAs start within 1 s of a junction** (92 % within 2 s), against **15.6 %** for random times. FA start − junction has a median of **+0.14 s**. Only 3 % of TPs start that close. Peak ratio median **54×** over 2 313 junctions (p10–p90 11–112×); ADL→ADL 41×, ADL→fall 76×, fall→fall 93×. My reference and window differ from the lead's, so "84×" is not a robust constant, but the order of magnitude holds. Decay to 3× the ADL median: median **5.1 s**, p90 9.8 s ("~8 s" is the long tail, not the typical case). |
| F4 junction detectable | **agree, and it is better than stated** | 0.9 | yes | Plain-jump variant (closest to the lead's): val block AUC **0.972**, P/R 0.86/0.71. Variance-normalised variant: val block AUC **0.993** (0.988 against all other blocks), event level at the train-chosen threshold: **val P 0.971, R 0.873, F1 0.92** at ±1.5 s, identical at ±0.5 s, so localisation is within one 0.2 s block. Recall by junction type (all streams): ADL→fall **0.982**, fall→ADL **0.992**, fall→fall 1.0, ADL→ADL 0.889 (the misses are where they do not matter). The 24 hard-cut junctions without a crossfade: recall 1.0. |
| F5 mat → radar | agree | 0.7 | no | Consistent with what I saw: the strongest-motion windows I picked per radar sit at low bins. I did not test the per-dataset mapping. |
| F6 annotation on the static change | agree | 0.7 | no | Matches README line 64 ("bounds placed from the change in the static CIR profile"). I did not recompute the AUCs. |
| F7 static fingerprint | partially | 0.7 | partly | Plausible, but "fall rec vs Autre/Empty ≈ chance" holds only for the *static amplitude* cue. The frame-timing cue (N1) separates them with val AUC **0.82**. |
| F8 D3 flagged | agree | 0.95 | yes (CSV) | D3 flagged 0.867 train / 1.0 val; val evaluable 67 (D1 33, D2 34). |
| F9 no activity context | agree | 0.75 | no | Generator shuffles segments (`build_streams.py`), so there is no mechanism for context. Not re-tested. |
| F10 prototype 0.884 | agree | 0.65 | no | Not re-run. The ranking mean (.948+.919+.785)/3 = 0.884 is consistent. With n = 67 falls the CI is wide (about ±0.04). Its junction/jump features would gain from the stronger normalised detector (F4 row). |

## New findings

### N1 — 26 % of frames are exact repeats; the repeat *pattern* fingerprints the corpus (including fall recordings vs Autre/Empty)
* The package aligns each radar to 50 Hz by **nearest frame** (`align.py`), and the native rate is about 37 Hz. As a result, **26.3 %** of frames repeat the previous frame exactly (`e_diff == 0`), and 84 % of repeat events happen on all 3 radars at once (shared clock).
* The spacing between repeats differs strongly by source corpus. In fall recordings a spacing of 3 frames almost never occurs (**2.8 %** of spacings, against **28.9 %** in ADL and 27.2 % in Autre). The spacing-2 fraction is 0.41 in fall recordings and 0.24 in ADL.
* A single feature, the fraction of spacing-3 repeats in a **2 s block**, separates fall recordings from ADL with **val AUC 0.80** (train 0.81) and from **Autre/Empty with val AUC 0.816** (train 0.818). That is exactly the pair the lead reported as chance (0.56) with the static fingerprint, so this is a complementary cue for the hardest negatives. The segment-level repeat rate is weaker (AUC 0.64–0.66).
* **Problems for our features:** (a) `e_diff` / frame-difference motion contains about 26 % structural zeros, so it is a noisy, aliased motion measure. Compute differences between *distinct* frames, or divide by the gap. (b) Freezes (≥ 3 consecutive repeated frames on all radars) are longer in fall recordings (median longest run 4 frames, max 73 = 1.5 s) than in ADL (median 1). 3.3 % of frames in the 3 s before D1/D2 falls are frozen, 1.3 % inside falls, 0 % in the 3 s after. 3.1 % of D1/D2 falls have more than 20 % of their frames frozen. These freezes look like "stillness" and can corrupt the onset localisation.
* **Risk:** this is a recording-pipeline artefact, not physics. It should transfer to the test set because the same generator and `align.py` build the test streams, but a model trained on it must still be validated subject-disjoint (done here: train → val AUCs agree). I recommend adding it as a feature and checking the ablation. It is a bonus, not a foundation.

### N2 — A variance-normalised two-sided complex change-point is a near-perfect junction/segment detector
* The statistic is ‖μ_after − μ_before‖² / (σ²_before + σ²_after) summed over 184 complex bins and 3 radars, on 1 s windows of `prof_02`. Normalising by the within-window variance suppresses real motion, which raises the spread on both sides, while a background swap does not.
* Val event-level results: **P 0.971 / R 0.873** (F1 0.92), against 0.86 / 0.71 for the plain jump. Block AUC 0.993. **98–99 % recall on the junctions that bound fall recordings**, localised within ±0.5 s.
* **Implication:** the segment model in the plan (junction → segment → localise the fall inside it) can rely on near-exact segment boundaries. Use this statistic, or feed it as a feature, for `t_since_jump`, `seg_len` and the junction-aware EMA reset. Since fall onset is 0–7.9 s after the junction (median 3.0 s), this also gives a strong prior for the start-error tie-break.

### N3 — I/Q is confirmed at sample level, so phase/Doppler features are valid
* The analytic (single-sideband) behaviour of z = I + jQ during motion (asymmetry 0.27–0.29 against a 0.078 null) means the **sign of the Doppler** (motion toward or away from the radar) is recoverable per range bin.
* For boundary precision (θ = 0.5), the end of a fall is where Doppler energy collapses to the noise floor on the mat bin. A phase-velocity feature (angle(z_t · conj(z_{t−1})) on distinct frames, see N1) should sharpen t_end. This is a proposal, not yet tested.

## Files
Scripts and JSON outputs: `/home/user/data_raw/agents/V2/` (`f1_iq_check.py`, `f1_offset_scan.py`, `f3_junction_ema.py`, `f4_junction_detect.py`, `n1_duplicate_frames.py`, `n1b_freezes.py`, plus `f1.json`, `f1_offset.json`, `f3.json`, `f4.json`, `n1.json`, `n1b.json`, `energy_baseline_val.csv`, `energy_eval.json`).

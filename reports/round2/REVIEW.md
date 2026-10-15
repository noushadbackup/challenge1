# Round 2 skeptical review of the FINAL system

Author: independent skeptical reviewer (round 2). I tried to refute FINAL's claimed score (`uwbfall/configs/final.json`; SR-CV 0.9221, val 0.9524). Every number below comes from code I ran in this session. All scores use the official `abc2027.evaluate` code with `--ignore-flagged`:
- val goes through the official CLI;
- SR-CV goes through the evaluator functions, either directly (my own concatenation of ground truth and durations) or through `srcv.scoring`.

My scripts and their outputs are in `work/REVIEW/` (copied from the scratchpad). The fresh-stream run keeps its regenerated CSVs, predictions and feature cache outside git in `/home/user/data_raw/review_fresh/`.

## TL;DR

* **The headline numbers reproduce exactly.**
  - SR-CV pooled ranking F1: FINAL 0.9221 (bags 0.9183 / 0.9235 / 0.9244); C3 0.8861.
  - Val (official CLI): FINAL 0.9524 / 0.9502 / 0.9524; C3 0.9646 / 0.9474 / 0.9495.
  - C3 → FINAL Δ +0.0361, CI [+0.020, +0.054], P = 1.000, 4/4 folds. I re-ran it with a different bootstrap seed and an independent pooled scorer.
* **I found no leakage that invalidates the SR-CV estimate.**
  - Fold train and fold test share 0 files and 0 subjects, checked independently from the composition CSVs. This covers fall, Autre/Empty and ADL.
  - The operating point comes from each bundle's own training OOF.
  - Nothing at predict time reads the composition or labels, and the transductive mask is off.
  - Two *mild* optimism sources remain (§2): round-1 hyperparameters were chosen on official-train OOF, which contains 18/24 of the SR-CV fall subjects and 6/8 of the ADL subjects; and about 50 variants were selected on the same 20 SR-CV streams.
* **Fresh regeneration test (new in this review).** I regenerated the same fold subjects with new, pre-registered composition seeds (9100+k): 20 streams and 253 evaluable falls that no round-2 decision ever saw. I predicted them with the existing bag-0 fold bundles.
  - **FINAL drops to 0.9079** (from 0.9183 on the original streams, bag 0). **C3 holds at 0.8881** (0.8851).
  - **C3 → FINAL still passes: Δ +0.020 [−0.002, +0.040], P = 0.966, 4/4 folds.** The gain shrinks by about 40% (+0.033 → +0.020): that is the winner's curse.
  - Per step:
    - the segment gate holds (+0.019, P = 0.995, 4/4);
    - the 4 s NMS halves (+0.008, P = 0.90, 3/4);
    - **the replay gate 0.1 → 0.3 reverses on its own** (C3 → C3G3 −0.0075, P = 0.014, 0/4).
  - Inside FINAL, gate 0.3 vs 0.1 is neutral on fresh streams (+0.002 for 0.1, P = 0.66) and favours 0.3 on the original streams and on val (0.9524 vs 0.9373). So **keep FINAL as is**.
  - The general lesson: the subject-cluster bootstrap does not capture composition-level variance. A "P = 1.000" for a boundary or post-processing tweak can be a property of 20 specific streams (§4).
* **Errors are lumpy, and that drives the test variance.**
  - One ADL subject, ADL_12, causes 10 of FINAL's 21 SR-CV false positives in bag 0, and 17 of the 60 false positives summed over 3 bags come from its *Laver_dents* recording alone.
  - Many of these FPs start within about 0.1 s of an ADL chunk start, i.e. at a junction. Over all FPs, 3.7 per bag start within ±0.5 s of a true junction, where there are 0 TPs.
  - Without ADL_12, pooled SR-CV is 0.935; with it, 0.922.
  - Whether the test draws an "ADL_12-like" subject matters more than any round-2 change. On the fresh streams the same recording again gives 11 of FINAL's 32 FPs.
* **Expected official test score: about 0.91** (central). 80% interval ≈ [0.85, 0.96]; 95% interval ≈ [0.80, 0.975]. The derivation is in §7. The val score of 0.952 is about the 83rd percentile of plausible test draws, because the val subjects are among the easiest in the pool.
* **Verdict.** Submit FINAL as frozen. I found no reason to change the config. The three risks with concrete mitigations are in §8. None of them calls for a code change before release, except an extra label-free monitor (R2).

## 1. What I re-ran

| check | how | result |
|---|---|---|
| SR-CV pooled F1, 3 bags × {C3, FINAL} | own script `indep_score.py`: concatenates the 4 fold `test_events.csv` files (`load_ground_truth(ignore_flagged=True)`), takes durations = last composition t_end, then `validate_predictions` + `evaluate` | identical to FINAL.md for every bag and fold: FINAL 0.9183 / 0.9235 / 0.9244, mean 0.9221; C3 mean 0.8861; 253 evaluable falls, 20 streams, 8.237 h |
| val, 3 bags × {C3, FINAL} | official CLI | FINAL 0.9524 (64/2/3), 0.9502, 0.9524; C3 0.9646 (65/0/2), 0.9474, 0.9495. The clean-run `srcv/results/final_b0/val.csv` is byte-identical to the scored file. |
| promotion C3 → FINAL | `srcv.promote`, bootstrap seed 12345 instead of 0 | Δ +0.0361, CI [+0.0200, +0.0537], P = 1.000, 4/4 folds |
| CI of FINAL | `bootstrap_ci`, seed 7 | [0.884, 0.954], bootstrap sd 0.0175 |
| fold integrity | own subject parser on all 8 fold-part composition CSVs + the official composition | 0 shared paths and 0 shared subjects between train and test in every fold. The stored `subject` column agrees 100% with my parser. Official train and val have disjoint ADL trailing numbers, which supports "trailing number = ADL subject". |
| fresh compositions (new) | seeds 9100+k, bag-0 fold bundles, official evaluator (§4) | FINAL 0.9079, C3 0.8881; C3 → FINAL +0.020, P 0.966, 4/4 |
| shipped bundle | `sha256sum -c models/final/MANIFEST.sha256`; config hash | 10/10 OK; `final.json` hash 204d32de158f1814 = bundle hash; trained on 18 streams; op (0.4, 3, 0.8, 0.6) |

## 2. Leakage audit

| suspected leak | finding | effect on the claimed score |
|---|---|---|
| Fold construction (subjects) | 4 folds, each holding out 2 D1 + 2 D2 + 2 D3 fall subjects with their own Autre/Empty files, plus 2 ADL subjects. I verified disjointness independently (§1). | none |
| ADL file reuse across fold train/test | 0 shared ADL paths. ADL reuse only happens *within* a fold part (adl_uses = 3 train, 2 test), as in the official splits. | none |
| Operating point chosen on test folds | No. `pipeline._finish` tunes (thr, smooth, min_dur, gap) on the bundle's own stream-grouped training OOF. The chosen points vary by fold: thr 0.3–0.4, smooth 1–5, min_dur 0.2–0.6. | none. But see §5: the leaky training OOF (ADL reused across OOF folds) under-counts FPs, so the operating point leans permissive. |
| Val used for tuning | Not in round 2: val is scored only as confirmation, and FINAL is not the val-best system (C3 / C3G3 score higher on val). Round 1 did look at val for a few candidates (E2: 2 candidates, E4: final only, INTEGRATION). | Val is not a clean test of round-1 choices. It is mildly optimistic for C3 and neutral-to-pessimistic for FINAL. |
| Round-1 hyperparameters (TCN E* = 8, w_tcn 0.75, prototype features, replay crop set, decode grid) | Chosen on the 13 official train streams' stream-grouped OOF. That data contains 18/24 of the SR-CV fall subjects and 6/8 of its ADL subjects, so the frozen choices have seen most SR-CV test subjects' data, though never as a subject-disjoint test. | Mild optimism of the absolute SR-CV level. I estimate ≤ 0.005–0.01, because these are few discrete knobs and most are on plateaus. Relative round-2 comparisons are barely affected. |
| Selection on the shared 20 SR-CV streams | About 50 variants were scored across R2_EVAL, I1, I2 and I3. Promoted: gate 0.3, the segment gate with floor 0.7, and 4 s NMS. The floor (0.5 vs 0.7) and the gate value 0.3 were looked at on SR-CV. | **Confirmed in part.** On fresh compositions FINAL is −0.010 (bag 0), the C3 → FINAL gain shrinks from +0.033 to +0.020, and gate 0.3 alone does not replicate (§4). |
| Transductive cues at predict time | `predict` reads only the stream's own radar data and features. `boundary.py` and `decode.py` never read the composition. `transductive.adl_duplicate_mask` is false. The drift monitors are cross-stream but do not change predictions. | none |
| Feature extractor fitted on all data | `uwbfall/extract.py` is stateless (EMA, block means). PCA and the models are fitted inside `fit` on fold-train only. | none |

## 3. Metric audit

I read `abc2027/evaluate.py` and checked the pipeline against it:
- **Ranking.** Mean F1 at θ 0.1/0.3/0.5, one-to-one matching, greedy by score with best-IoU. `srcv.scoring` calls exactly these functions, and my independent concatenation gives identical counts.
- **Ignore zones.** Test GT carries `ignore = flagged` (generator split `test`), and SR-CV test events do the same. An unmatched prediction that overlaps an ignore zone ±2 s is neither TP nor FP, so SR-CV scoring is the test rule.
- **Durations.** T/50 from the composition equals what `stream_durations` reads from the h5 (srcv V3/V6). My scorer uses the last composition t_end, and the totals agree (8.237 h).
- **θ.** Nothing in the pipeline assumes a single θ. The operating point maximises the ranking F1 itself (mean over 0.1/0.3/0.5) on training OOF.
- **Tie-breaks.** Median start error at 0.3 is 0.04 s on SR-CV and val, from the replay of `propose_fall`. False alarms per hour at 0.3 is 2.43 on SR-CV and about 1/h on val.

**No metric mistake found.** One subtlety for interpretation: SR-CV pooled F1 is a micro-average over 20 streams. The official test pools 5 streams, so a single test draw is noisier than the pooled number (§7).

## 4. Fresh SR-CV regeneration (reviewer test, pre-registered)

**Why.** Every round-2 decision used the same 20 SR-CV test streams, and about 50 variants were looked at. The subject-cluster bootstrap resamples *subjects*, not *compositions*: which ADL chunk sits next to which fall, the crop margins, the junction positions. Boundary and post-processing tweaks act on exactly those, so their P values can be over-confident.

**Design.**
- Same 4 fold subject splits and the same 253 evaluable falls (fall_uses = 1, so the same fall files).
- Test parts re-planned with the official `prepare_split("test")` + `finish_plan` and **new seeds 9100+k**, chosen before looking at any result (`work/REVIEW/fresh.py`). The streams are materialised in memory, and only features are cached, in `/home/user/data_raw/review_fresh/cache`.
- Predictions use the **existing bag-0 fold bundles**, so there is no retraining:
  - C3 = `srcv/runs/uwb_default/fold_k`;
  - FINAL = `srcv/runs/final_b0/fold_k`;
  - C3G3 = the C3 bundle with the boundary gate overridden to 0.3, as `predict --boundary`;
  - N4 = C3G3 plus `clip.nms_time(4.0)` per stream, I1's definition.
- Scoring uses the official evaluator via `srcv.scoring` with the fold directory pointed at the new CSVs, and `srcv.promote` with 5000 subject-cluster resamples. This is one bag only, so it is a confirmation, not a 3-bag promotion.

**Pooled ranking F1, bag 0 (the deployed seed set):**

| system | original 20 streams | **fresh 20 streams** | fresh TP/FP/FN @0.3 | fresh folds 0/1/2/3 |
|---|---|---|---|---|
| C3 (round-1 integration) | 0.8851 | **0.8881** | 239/38/14 | .897/.825/.889/.946 |
| C3G3 (replay gate 0.3) | 0.8889 | **0.8805** | 237/40/16 | |
| N4 (+ 4 s NMS) | 0.9044 | **0.8885** | 236/34/17 | |
| **FINAL = G7N4 (+ segment gate)** | 0.9183 | **0.9079**, CI [0.841, 0.959], FA/h 3.88 | 240/32/13 | .921/.842/.894/.980 |
| FINAL with replay gate 0.1 (FG1) | 0.9113 | 0.9100 | | (val 0.9373, 63/3/4) |

**Promotion chain on the fresh streams vs the original streams (bag 0):**

| step | original Δ (bag 0) | **fresh Δ [95% CI]** | fresh P(Δ>0) | fresh folds | verdict |
|---|---|---|---|---|---|
| C3 → C3G3 (gate 0.1 → 0.3) | +0.0038 (P 0.84, 2/4); 3-bag +0.0082, P 1.000 | **−0.0075 [−0.016, 0.000]** | 0.014 | 0/4 | does not replicate |
| C3G3 → N4 (time NMS) | +0.0155 | +0.0080 [−0.003, +0.021] | 0.899 | 3/4 | replicates, about half the size |
| N4 → FINAL (segment gate) | +0.0139 | **+0.0195 [+0.004, +0.036]** | 0.995 | 4/4 | replicates |
| C3G3 → FINAL | +0.0294 | +0.0274 [+0.009, +0.046] | 0.997 | 4/4 | replicates |
| **C3 → FINAL** | +0.0332 | **+0.0199 [−0.002, +0.039]** | **0.966** | **4/4** | **passes the rule again** |
| FINAL → FG1 (gate back to 0.1) | −0.0070 (P 0.07, 1/4) | +0.0021 [−0.007, +0.012] | 0.658 | 1/4 | neutral; keep 0.3 |

**Reading.**
- FINAL's absolute level drops by about 0.010 on unselected compositions. Part of this is composition noise: ADL_12's *Laver_dents* chunks give 11 FPs here, against fewer on the original streams. Part is the winner's curse.
- The FP reduction C3 → FINAL is smaller: 38 → 32, against 36 → 21 on the original bag 0.
- Recall is unchanged: 240 vs 239 TP.
- The segment gate is the most robust round-2 change.
- The replay gate 0.3 was promoted with P = 1.000. Its effect is in fact composition-dependent and about zero in expectation. Within FINAL it is neutral to slightly positive, so no change is warranted.
- **Recommendation for any future round:** confirm promoted post-processing changes on a second, fresh composition before freezing them. A subject bootstrap on one composition is not enough.

## 5. Fragile thresholds

* **Score-cut plateau.** I dropped FINAL rows below a score cut, as a proxy for raising the decode threshold, and scored with the official evaluator (3-bag mean):

  | cut | none | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.70 |
  |---|---|---|---|---|---|---|---|
  | SR-CV | 0.9221 | 0.9241 | 0.9242 | 0.9250 | 0.9268 | 0.9168 | 0.8761 |
  | val | 0.9517 | 0.9517 | 0.9517 | 0.9562 | 0.9586 | 0.9452 | 0.9147 |

  - The frozen operating point sits on a plateau that extends upward to about 0.55; FINAL leaves at most +0.005 on the table, in hindsight.
  - There is a **cliff above about 0.6**. If test probabilities come out about 30% lower than on SR-CV, recall collapses. A writer or domain shift that dampens motion energy would do this.
  - I do **not** recommend changing the cut: the gain is hindsight on the selection streams, as in I3.
* **Segment-gate floor.** The gate multiplies p by 0.7 + 0.3·P(fall clip), so a broken segment classifier can at worst damp every probability by 0.7.
  - Proxy: N4 (no gate) with a score cut at 0.4/0.7 = 0.571 gives SR-CV 0.9166 and val 0.9606.
  - So even an adversarial gate failure costs about 0.005 on SR-CV, as long as the probabilities themselves stay calibrated. The floor makes the gate a low-fragility component.
  - Skeptic's corollary: most of the gate's SR-CV gain over N4 (+0.014) is reproduced by a plain score cut chosen in hindsight (+0.0086). Its *segment-discriminative* contribution is therefore about +0.005; the rest is an effective threshold raise.
  - That is consistent with the leaky training OOF (ADL re-use, OOF FA/h 1.0–1.9 vs SR-CV 2.4) choosing a too-permissive operating point. The gain is still real on held-out subjects, so this does not argue against FINAL.
* **Operating point across folds.** (thr, smooth, min_dur) vary across fold bundles and bags: thr 0.3–0.5, smooth 1–5, min_dur 0.2–0.6. The OOF grid surfaces are flat (top cells within about 0.01), so the instability costs little.
* **Junction-onset rule (tested and rejected, new).** Evaluable falls always start ≥ 1.00 s after their true junction (min 1.00, verified on all 253). In hindsight, 3.7 FPs per bag start within ±0.5 s of a *true* junction, with 0 TPs there.
  - I pre-registered one deployable rule: drop a final row whose start lies within ±0.6 s of a *detected* junction peak (fold bundle's junction threshold).
  - It **fails badly**: Δ −0.0128, P = 0.19, 1/4 folds, val 0.952 → 0.936. It drops 78 rows over 3 bags × 5 targets, most of them TPs.
  - The junction detector fires on the falls themselves (a fall changes the static profile). **Do not add an onset ≥ junction + 1 s rule unless the junction is known to be a real clip boundary.**

## 6. Dependence on generator artefacts

What FINAL relies on, and how it degrades if the official `rebuild_test.py` behaves differently from the public `_write_stream`:

| artefact | used by | if it is absent or different on test | evidence |
|---|---|---|---|
| 0.2 s crossfade junctions / EMA junction spike | prototype HGB features (EMA junction, t_since/t_until), segment gate (junction-to-junction segments), `det*` replay crops | fewer or different segments; gate floor bounds the damage; HGB features shift | physics-only fallback (no junction/segment/fingerprint features): val 0.9239 vs 0.9524 (FINAL.md §5) |
| Sample-and-hold repeat spacing (spacing-3 share about 0.025 in fall recordings vs about 0.25 in ADL) | segment classifier features `rep_rate`, `rep_sp2/3/4`, `rep_n` | P(fall clip) would drop on true fall clips, so the gate damps them (worst case ×0.7, §5) | I1: segment OOF AUC 0.975 with vs 0.928 without the repeat features |
| GT = `abc2027.annotate.propose_fall` on the source recording | replay boundaries (median start error 0.04 s, tie-break 1) | lose the boundary gain: about −0.015 rank F1 (C2 → C3G3 +0.0146) and the start-error tie-break | EVAL.md |
| ADL reused bit-exactly within the test split | not used (mask OFF) | — | — |

The test fall corpus is the same LIARA corpus, from new subjects, presumably aligned by the same `align.py`, so the repeat and crossfade cues should transfer (DATA_TEAM_VERDICT caveat). The drift monitors cover global repeat share, junction rate and PSI. **They do not cover the per-segment spacing-3 cue that feeds the gate** (R2 below).

## 7. Expected official test score

**Test-draw simulation** (`testdraw.py`):
- Takes per-subject official TP/FP/FN counts on SR-CV for each bag (`srcv.scoring.cluster_counts`; the totals equal the official counts).
- Draws 20 000 test-shaped splits: 2 D1 + 2 D2 + 2 D3 fall subjects and 2 ADL subjects, without replacement, plus a random bag.
- Computes the pooled ranking F1 of each draw.

| system | mean | sd | 2.5% | 10% | 25% | 50% | 75% | 90% | 97.5% | with an evaluable-D3 subject (P3_3, p = 0.25) | without |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FINAL | 0.9215 | 0.032 | 0.849 | 0.877 | 0.902 | 0.925 | 0.945 | 0.960 | 0.973 | 0.931 | 0.918 |
| C3 | 0.8856 | 0.045 | 0.781 | 0.824 | 0.859 | 0.892 | 0.918 | 0.937 | 0.957 | 0.902 | 0.880 |

With the fresh-stream counts (`testdraw_fresh.py`):
- FINAL bag 0 on fresh streams alone: mean 0.909, sd 0.052, 2.5–97.5% [0.79, 0.98]. ADL_12 has 15 FPs there, which fattens the lower tail.
- The 3 original bags plus fresh, pooled as 4 replicates: mean 0.919, sd 0.038, [0.82, 0.98].

If the test holds only one fall subject per dataset (see R3), FINAL's mean is 0.905, sd 0.055, 95% [0.77, 0.985] (`testdraw111.py`).

**Adjustments to the central value** (starting from the original-stream mean 0.922):

| item | adjustment |
|---|---|
| round-2 selection on the same 20 streams: FINAL bag 0 drops 0.9183 → 0.9079 on fresh compositions (§4) | −0.010 |
| round-1 hyperparameters saw most SR-CV subjects (§2) | −0.005 |
| final bundle trained on 24 fall + 8 ADL subjects instead of 18 + 6 | +0.003 (small; not measured) |
| test writer or annotation differs (tail risk, not in the central value) | −0.015 to −0.03 if it happens |

**Estimate: 0.91 (1 sd ≈ 0.04). 80% interval [0.85, 0.96]; 95% interval [0.80, 0.975].**
- The lower tail is driven by the chance of an ADL_12-like subject.
- The upper end is driven by a val-like easy draw.

For calibration:
- Val at 0.952 is the 83rd percentile of the FINAL draw distribution.
- Val's ADL subjects (ADL_0, ADL_11) produce 0 and 1 FP on SR-CV, and its fall subjects are among the cleanest.
- The prototype's history fits the same pattern: val 0.884 vs rotation mean 0.861.

**Relative to C3**, the ranking decision is robust even if the absolute level is not:
- In paired draws (`testdraw_pair.py`, same subjects and same bag), FINAL > C3 in 96.0% of draws, mean +0.036, 5th percentile +0.002.
- On fresh compositions the gain is +0.020, P = 0.966, 4/4 folds.

## 8. Top 3 risks and concrete mitigations

**R1. A false-alarm-prone unseen ADL subject (largest, most likely).**
- On SR-CV, FPs are concentrated: ADL_12 alone accounts for 10/21 FPs on bag 0. Its *Laver_dents* recording gives 17 of the 60 FPs summed over 3 bags, many of them rows that start at the ADL chunk junction (fpact.py, junc.py).
- Test has 2 unseen ADL subjects, and adl_uses = 2 means every part of an ADL recording appears about twice, so one bad recording produces correlated FPs. On the 20-stream pool, ADL_12 costs 0.013 (0.935 → 0.922). In test-shaped draws, the draws containing ADL_12 (p = 0.25) average **0.886**, against 0.933 without it (`td_adl12.py`).
- Mitigations:
  1. Keep the operating point as is. Raising it trades recall for precision with only +0.005 headroom, and the cliff starts at 0.6 (§5).
  2. A *rules-cleared* ADL duplicate mask would remove exactly this failure, since bit-exact re-use only happens for ADL. The lead should ask the organisers now (PLAN §5) rather than decide on release day. If allowed, apply it only to predictions not overlapping a high-P(fall clip) segment.
  3. Do not add a junction-onset filter (§5: it fails, P = 0.19).

**R2. The test writer or alignment changes the structural cues (crossfade, sample-and-hold spacing) or the annotation.**
- The gate floor limits the damage from the segment classifier to about −0.005 (§5). The HGB junction features and the replay gain (+0.015 and the start-error tie-break) are larger exposures.
- Mitigations:
  1. Add one label-free monitor before release: the share of junction-to-junction segments with `rep_sp3` < 0.1, on test vs the training reference. It should be about the fall-clip share. Also add the mean P(fall clip) over segments containing a candidate with p > 0.6, vs training OOF. Both are cheap reads of `stream_probs` parts.
  2. Keep the existing decision rule: switch to the physics fallback (val 0.924) only on structural ALERTs.
  3. On release day, eyeball 3–4 predicted intervals per stream against the motion energy before submitting.

**R3. Test shape and D3 composition differ from SR-CV.**
- Subject ids in the package skip P1_4, P2_4, P3_4 and ADL 1, 2, 3, 7, 8. If those are the test subjects (one per dataset), the test has 3 fall subjects and higher variance (sd 0.055).
- The evaluable-D3 behaviour then rests on a single subject. SR-CV has only one evaluable D3 subject (P3_3, recall 0.958, peak at bins ~61), so recall on an evaluable D3 subject at bins ~42–48 has never been measured.
- Mitigations:
  1. No model change: I2's D3 hedges all failed the rule.
  2. On release day, report the test-side count of predictions on R29171 far bins as a monitor.
  3. Accept the variance. It is a property of the draw, not of the system.

## 9. Smaller findings

* Training-OOF FA/h (1.0–1.9) is roughly half of SR-CV's (2.4). Every OOF-chosen knob is therefore biased toward permissiveness. A subject- and ADL-file-grouped inner OOF for the operating point would be the principled fix. I3's re-selection on SR-CV OOF showed no extra gain on top of G7N4, so this is not urgent.
- The val drop C3 → FINAL (bag 0: 65/0/2 → 64/2/3) is two events: a post-fall row at val_02 ≈ 314 s and a widened replay interval at val_03 (I1 §4). Both are within val noise. On fresh streams C3 → FINAL still improves in 4/4 folds, so the val drop is not a sign of a systematic regression.
- The deployed bundle is a single TCN bag. The bag sd on pooled SR-CV is 0.003, and per test draw it is part of the 0.032 above.
- I could not refute bit-exact reproducibility: the manifest verifies, and the config hash matches.

## 10. Files

`work/REVIEW/` (small, in git):
- `indep_score.py`: independent pooled scorer.
- `testdraw.py`, `testdraw111.py`, `testdraw*.json`: test-draw simulation.
- `fpact.py`, `junc.py`: FP attribution and the junction-offset diagnostic.
- `detj.py`, `jrule.py`, `promote_FINAL_vs_JR0.6.json`: detected junctions and the junction rule.
- `cut.py`: score-cut plateau.
- `fresh.py` (gen / pred / score), `fresh_results.json`: fresh regeneration, C3 and FINAL.
- `fresh2.py`, `fresh_chain.json`: the C3G3 / N4 decomposition on the fresh streams.
- `fresh3.py`, `run3.sh`, `score3.py`, `fg1_results.json`, `val_FG1.json`: FINAL with replay gate 0.1, on fresh, original and val.
- `testdraw_fresh.py`, `testdraw_pair.py`, `td_adl12.py`: draw simulations with fresh counts, paired FINAL vs C3, and with/without ADL_12.
- `fpact_fresh.py`, `junc_fresh.py`: FP attribution and junction offsets on the fresh streams.
- `val_*.json`: official CLI reports.

The scripts were run from the scratchpad. Relative inputs such as `detj.json` are expected in the working directory, so run them from `work/REVIEW/`. All Python needs `PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src`. Fresh-stream predictions need `UWBFALL_CACHE=/home/user/data_raw/review_fresh/cache` to reuse the cached features.

Outside git: `/home/user/data_raw/review_fresh/` holds fold CSVs with seeds 9100+k, predictions (`preds/{C3,C3G3,N4,FINAL}`, `preds_FG1/{fresh,orig,val}`), the feature cache (about 0.5 GB, deletable) and the logs.

Compute: about 1 h wall for this review, on ≤ 2 threads, shared with another agent. No model was retrained. The package and the frozen config were not modified, and nothing was committed.

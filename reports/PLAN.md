# ABC 2027 UWB Fall Challenge — Plan to win (v2, after the 20-agent team review)

Evidence: `reports/REPORT.md` (initial analysis), `reports/team/DATA_TEAM_VERDICT.md` (10-person data
team: 8 voters + 2 leads), `reports/team/expert_X1..X5.md` (field experts), `work/E1..E5/README.md`
(executors), `work/INTEGRATION/` (combination check). All scores = official evaluator, `--ignore-flagged`.

## 0. Where we stand

| system | val ranking F1 | F1 @0.1/0.3/0.5/0.7 | TP/FP/FN @0.3 | med. start err | source |
|---|---|---|---|---|---|
| official energy baseline | 0.315 | .358/.338/.249/.060 | 34/100/33 | 0.32 s | package |
| prototype (HGB, 0.2 s blocks) | 0.884 | .948/.919/.785/.504 | 62/6/5 | 0.22 s | analysis/ |
| prototype + annotator replay (E2, config B) | 0.928 | .948/.933/.904/.726 | 63/5/4 | 0.06 s | work/E2 |
| TCN, 3-seed bag (E4) | 0.930 | .970/.970/.851/.507 | 65/2/2 | 0.34 s | work/E4 |
| 0.75 TCN + 0.25 prototype (E4) | 0.934 | .985/.970/.848/.561 | 64/1/3 | 0.33 s | work/E4 |
| **0.75 TCN + 0.25 prototype + replay B** | **0.965** | .985/.985/.924/.818 | **65/0/2** | **0.06 s** | work/INTEGRATION |
| TCN alone + replay B | 0.925 | .970/.925/.881/.776 | 62/5/5 | 0.045 s | work/INTEGRATION |

**Caveats (agreed by the data team and X4/X5):**
* Val has 67 evaluable falls from 4 subjects (no evaluable D3), so differences below about **0.03–0.05 are noise**.
  The pseudo-test bank (E1: 50 regenerated streams) puts the prototype at **0.877 ± 0.020**; val 0.884 is a typical draw.
* Train OOF grouped by stream is **leaky** (ADL files are reused bit-exactly across streams; fall subjects repeat).
* Replay helped the blend (+0.030) but slightly hurt the bare TCN (−0.005). This interaction is unresolved, so the
  0.965 is a candidate, not a promoted result, until it passes the SR-CV rule (§2).

## 1. What the team changed in the analysis (see DATA_TEAM_VERDICT.md)

* Confirmed: I/Q format (8/0/0 votes), stream timing structure, junction artefact, random activity context, Dataset 3 mostly ignored.
* Corrected: the EMA junction spike is ~55×, not 84×. The "motion peak before onset" claim was itself an EMA artefact:
  junction-immune motion peaks *inside* the fall for 87–90 % of D1/D2 falls. D3 *is* seen, by R29171 at bins ~40–68.
  The fingerprint AUC vs ADL is 0.94–0.98.
* New, confirmed:
  1. **The ground truth is produced by the package's own `abc2027.annotate.propose_fall`**, and flags are deterministic.
     Replaying it on the detected crop (on distinct frames) is the best boundary method found.
  2. **23–27 % of frames are sample-and-hold repeats.** The spacing-3 repeat share separates fall clips from
     Autre/Empty (segment AUC ≈ 1.0).
  3. **ADL content is reused bit-exactly within a split**, while fall/Autre/Empty content never repeats.
  4. Evaluable falls start ≥ 1 s after their junction and last ≥ 1 s; hard cuts are ignore zones; durations spike at exactly 2.00 s.
  5. Chute04 (falling while sitting down on a chair) is the boundary bottleneck.

## 2. Validation protocol (replaces v1 §2) — X3, X4, X5 P0

1. **SR-CV — subject-disjoint regenerated CV.** Pool the train+val subjects and build 4 folds shaped like the test
   (each holds 2 D1 + 2 D2 + 2 D3 fall subjects with their own Autre/Empty files, plus 2 ADL subjects, i.e. ADL file ids
   by trailing number). Regenerate each fold's held-out streams with the official generator at test settings
   (`adl_uses=2`, 0.2 s fade, flagged → ignore), using E1's verified writer, extractor and scorer.
2. **Promotion rule:** keep a change only if (a) a paired subject-cluster bootstrap on pooled SR-CV OOF gives
   P(Δ ranking F1 > 0) ≥ 0.90, (b) ≥ 3 of 4 folds improve, and (c) holdout val does not drop by more than its noise.
   Seed-average every candidate (≥ 3 seeds) before comparing.
3. **Determinism:** HGB `early_stopping=False`; LightGBM `deterministic=True, force_row_wise=True`; fixed fold
   artefacts shared by all models (including stacking).
4. **Pre-registered final selection:** the best seed-averaged SR-CV system; val and the E1 bank only confirm it.

## 3. Pipeline v2

```
raw h5 ─► S0 features: complex I/Q, distinct-frame motion, 0.2 s static profile, per-radar zones
       ─► S1 junctions: variance-normalised two-window statistic, threshold set for fall-junction recall (q0.95 ≈ 0.95)
       ─► S2 segment classifier (LightGBM): fall clip / ADL / Autre-Empty
              features: static change, motion burst, fingerprint, spacing-3 repeat share, segment length
       ─► S3 framewise fall probability: 0.75 · TCN (3 seeds) + 0.25 · GBDT (HGB/LightGBM/XGBoost/CatBoost mean)
       ─► S4 decode: hard 1-D NMS, at most ONE event per fall-clip segment; threshold maximises mean F1(θ .1/.3/.5) on SR-CV
       ─► S5 boundaries: annotator replay (propose_fall on distinct frames), median of 6 short crops,
              gate tIoU ≥ 0.1 vs the detector, 1.0 s duration floor (extend end only), no start padding
       ─► S6 score = event-level hit model (LightGBM on peak prob, replay agreement, crop spread, segment-class prob)
       ─► CSV (+ validate_predictions)
```

**Models (all open-source):** PyTorch dilated TCN (MS-TCN++ style, main member); LightGBM, XGBoost, CatBoost and
sklearn HGB (committee); `abc2027.annotate.propose_fall` (package code, boundaries); LightGBM segment classifier
and event hit model; ruptures (fallback boundary only). ResNet-18, ActionFormer, TriDet and MiniRocket move to optional
P2 probes, because they are too heavy for the CPU budget without measured gain (X5).

**Multi-agent / voting, as measured (E5):**
* The model committee (soft vote of 4 GBDTs) gives a small gain over one GBDT (+0.015 OOF, not significant).
* "≥ 2 radars agree" voting **loses recall**. Use anchor-radar logic instead (X1): pick the radar with the largest
  static change per segment and use the others as context.
* Stacking, weighted segment fusion and soft-NMS did **not** beat a plain weighted soft vote.
  The decoder is hard NMS with one event per clip (X2).
* The physics-only model is kept as a **switch** (fallback triggered by drift monitors), not as a blend member (X4).

## 4. Work packages, next round

| # | package | owner role | exit criterion |
|---|---|---|---|
| W1 | SR-CV folds + regenerated held-out streams + promotion-rule script | E1 + X4 | prototype, TCN and blend scored on SR-CV with CIs |
| W2 | Re-validate blend + replay on SR-CV; fix the TCN-alone/replay interaction (gate on TCN interval width, replay-agreement feature) | E2 + E4 | promotion rule passes |
| W3 | Clip-level decoding (one event per fall clip) + S2 segment classifier + post-fall negative class (X3) | E5 | fewer FPs/fragments on SR-CV |
| W4 | Chute04 boundaries; D3 hedge (R29171 bins 40–68 zone features); flagged-event loss masking | E3 + X1/X3 | per-type hit@0.5 up, D3 recall on P3_3 measured |
| W5 | Event hit model for the score column + operating point | E5 + X4 | stable threshold across folds |
| W6 | Submission package: `uwbfall` CLI (extract/train/predict from raw test h5), run registry, determinism, Makefile, drift monitors, validate_predictions | X5 | one command reproduces val CSV bit-exactly |

## 5. Rule-compliance ledger (X5)

| technique | status |
|---|---|
| Package data, `segments/` and generator, used to regenerate training/validation streams | allowed (shipped in the package) |
| `abc2027.annotate.propose_fall` at inference | package code, allowed; documented in the submission |
| Distinct-frame / spacing-3 repeat features (within one stream) | allowed; a property of the provided streams |
| Cross-stream ADL duplicate mask (transductive across test streams) | **default OFF**; ask the organisers before using |
| External LIARA recordings | prohibited, not used |
| Pretrained weights | none used |

## 6. Risks

| risk | mitigation |
|---|---|
| Test generator or writer differs (repeat structure, junction behaviour) | drift monitors on test + physics-only switch; replay fallback to stream-grid motion (≈ +0.02 instead of +0.04) |
| Val luck (67 events) | SR-CV + bank + promotion rule; pre-registered selection |
| D3 subject like P3_3 in the test (evaluable D3) | W4 D3 hedge |
| Invalid submission file | `validate_predictions` in CI on `streams/test` ids |
| Code verification for top teams | W6 package, run registry, deterministic seeds |

## Change log
* v1 — initial plan from the analysis (val 0.884 prototype).
* v2 — team review. Added SR-CV and the promotion rule; annotator-replay boundaries (E2: +0.044 val);
  TCN as main detector (E4: 0.930); integration 0.965 val (candidate); multi-agent voting re-scoped to what measured
  well; heavy deep models moved to P2; compliance ledger and packaging pulled forward.
* v3 — round 2 done (`reports/round2/`): uwbfall end-to-end package; SR-CV harness; FINAL = 0.75 TCN + 0.25 HGB
  + annotator replay (gate 0.3) + segment-classifier gate + 4 s NMS. SR-CV 0.922 (fresh-composition re-check 0.908),
  val 0.952, 2.43 FA/h. Final models trained on train+val (`models/final/`); audit PASS (fresh clone reproduces
  bit-exactly, deterministic, rule-compliant); release-day steps in `SUBMISSION.md`.

# Expert review X3: fall detection and ambient-assisted-living sensing

Reviewer: X3 (radar fall-detection literature, the LIARA papers, realistic failure modes). Scope: PLAN.md v1, read alongside REPORT.md, DATA_TEAM_VERDICT.md and expert_X1.md.

**Verdict: sound with changes.** The detector design is close to right. What needs changing:

- **The validation protocol measures the wrong kind of variance.** On test, the score depends on which unseen subjects appear, not on crop seeds.
- **The plan's D3 assumption is unsafe.** "D3 is ignored, so don't optimise for it" does not hold.
- **The remaining errors are localisation errors inside fall clips, not detection errors.** They should be handled with segment-level decoding plus X1's annotator replay.

Every number below comes from scripts I ran in this session, in `/home/user/data_raw/agents/X3/`:

| script | what it does |
|---|---|
| `x3_subject.py` | per-subject and per-FP breakdown, using the official `_match_stream` |
| `x3_proto_checks.py` | reproduces the prototype; one-event-per-segment rule; held-out-subject recall |
| `x3_d3.py` | D3 range-gated feature AUC |
| `x3_zone.py` + `x3_proto_checks_lib.py` | mat-zone features |
| `x3_zone_lib.py` + `x3_zone2_run.py` | zone-agnostic features, flagged blocks dropped |

Labels used below:

- *(estimate)*: a number I did not measure.
- *(inference)*: a deduction from file names or papers, not from data.

---

## 1. What the fall-detection view adds (measured)

### 1.1 Every remaining val error is a localisation error inside a fall clip, except one
Prototype `val_full.csv`, official matcher at θ=0.3, after the ignore-zone rule:

- **6 FPs and 5 FNs.** 5 of the 6 FPs lie inside fall-recording segments. Each sits in the same clip as that clip's own unmatched fall, so they are FP+FN pairs.

| FP | clip | what happened |
|---|---|---|
| val_00 809.2–812.0 | Chute01, D1 | over-wide/late; tIoU 0.29 with GT 807.72–810.46 |
| val_01 1234.6–1239.2 | Chute04, D2 | over-wide (4.6 s for a 1.22 s fall); tIoU 0.27 |
| val_01 1455.6–1456.8 | Chute03, D1 | **fragment**: a second prediction for a fall already hit by 1453.8–1454.8 |
| val_02 315.6–317.0 | Chute04, D1 | **post-fall motion**: fires 3.7 s after GT end (309.9–311.9) |
| val_02 1305.4–1306.2 | Chute04, D1 | **post-fall motion**: fires 0.56 s after GT end (1302.84–1304.84) |
| val_03 741.4–743.2 | Autre, D2 | the only real false alarm; 1.7 s after the junction into an Autre clip (junction residue) |

- **Exactly one fall per fall clip.** Every one of the 478 fall-recording segments contains exactly one annotated fall, and no event starts outside a fall clip. The generator guarantees this structure (`fall_uses: 1`, one Chute file per segment). The plan does not exploit it.
- **The failure modes match the classic radar fall-detection ones:**
  - post-fall movement on the floor (adjusting, attempts to get up), which fires *after* the fall;
  - multi-phase chair falls (Chute04, consistent with "falling while trying to sit"), which produce over-wide or late segments.
- **Chute04 hit@0.5 per val subject** is 0.25 / 0.60 / 0.67 / 0.00 (P1_8 / P1_9 / P2_8 / P2_9).

### 1.2 Subject-level variance dominates, and the plan's pseudo-test bank cannot see it
- **Subject inventory.** Each dataset has fall subjects 1, 2, 3, 5, 6, 7, 8, 9, with 20 falls each (5 per Chute type). The LIARA JBHI paper reports **10 participants**. So the test set probably holds subjects 4 and 10 of each mat *(inference)*: about 2 subjects per mat and about 120 falls over 5 streams, the same shape as val (6 subjects, 119 falls, 5 streams).
- **Evaluable falls per subject vary a lot**: from 6 (P2_2) to 18 for D1/D2 subjects, and for D3 either 0 or 16 (only P3_3).
- **Held-out-subject recall.** I removed one subject's segments from training and scored only that subject's falls on train streams, with the same features and the val post-processing (0.4, 3, 0.8, 0.6):

| held-out subject | recall θ0.1 | θ0.3 | θ0.5 |
|---|---|---|---|
| Test_P1_5 (D1) | 12/18 | 12/18 | 11/18 |
| Test_P2_3 (D2) | 18/18 | 18/18 | 17/18 |
| **Test_P3_3 (D3, the only evaluable D3 subject)** | **0/16** | **0/16** | **0/16** |

  - ADL files are still shared with training, so these figures are optimistic for the ADL side. They are not optimistic for the falls.
- **Val per subject (prototype):** recall @0.3 is 0.93 / 0.89 / 0.94 / 0.94, and @0.5 is 0.80 / 0.67 / 0.89 / 0.81.
- **Val subject pairs.** If test held one D1 and one D2 val subject (FPs pro-rated to 50 %), ranking F1 would range 0.873–0.894.
- **Start-error bias depends on the subject.** Median signed start error is −0.13, +0.02, +0.08 and +0.22 s. Tie-break 1 is therefore subject-dependent, and a global offset tuned on 4 subjects will not transfer.

**Consequence:** the plan's bank of ≥50 regenerated *val* streams re-uses the same 67 falls from the same 4 subjects. Its sd measures crop and neighbour noise and will under-state the test variance badly. LIARA's own papers evaluate leave-one-subject-out: Maitre et al., JBHI 2021 ([constellation.uqac.ca/6269](https://constellation.uqac.ca/6269/)), and Abudalfa & Bouchard 2024 ([easychair hK22](https://easychair.org/publications/preprint/hK22)).

### 1.3 D3 is not free, and the current model is blind to an unseen evaluable D3 subject
- **PLAN §5 is wrong as written.** It says D3 is "ignored on test → don't optimise for it; predictions there are free". Two problems:
  - Flags are per-event and subject-dependent: P3_3 is 20 % flagged, 16 evaluable falls.
  - Predictions are free only within ±2 s of a *flagged* event (L2's N5 correction).
- **Test exposure.** If a test D3 subject resembles P3_3, test gains about 16 evaluable D3 falls. At 0/16 recall, that alone costs about 0.08–0.10 ranking F1 *(estimate, from F1 = 2TP/(2TP+FP+FN) with ~70 D1/D2 falls)*. Naively, with 1 of 8 known D3 subjects being "evaluable-type", P(≥1 of 2 test D3 subjects) ≈ 23 % *(estimate)*. The expected loss is about 0.02–0.03, with a fat tail.
- **The signal exists, but only in the right range gate.** On the 16 unflagged P3_3 falls against 2400 ADL 2 s windows, all on R29171:

| feature | AUC |
|---|---|
| static change (1 s before/after) over all 184 bins (the prototype's form) | 0.740 |
| **static change restricted to bins 40–68** | **0.994** |
| frame-diff energy, all bins | 0.953 |
| mot_map energy, bins 40–68 | 0.932 |

- **Adding features alone does not fix it, because the training labels for D3 are almost all flagged/weak:**

| variant | P3_3 held-out recall | P1_5 held-out recall | val ranking F1 |
|---|---|---|---|
| + per-radar mat-zone features (`x3_zone.py`) | 2/16 | 15/18 | 0.868 (−0.016, post-proc not retuned, within noise) |
| zone-agnostic "max over zones" + all flagged fall blocks dropped from training (`x3_zone2_run.py`) | 0/16 | — | **0.797** |

  - The second variant confounds two changes. Dropping flagged *fall blocks* while keeping their neighbouring blocks as negatives is the likely cause of the drop. **Keep flagged falls as positives, or drop their whole clip.** Never drop only the fall interval.

### 1.4 One-event-per-segment needs a proper junction detector
- I applied "keep only the highest-scoring event per detected segment", using the prototype's own jump peaks (p97 of `jump_max`). Val stayed **0.884 → 0.884**.
- The reason: the fall's own static change produces a jump peak, so the crude detector splits each fall clip at the fall.
- With oracle segments, the rule removes only the 1 fragment FP: about +0.004 ranking F1 (computed from the confusion counts).
- Its real value is as the frame for decoding: one clip, one fall, localised by the replay. Section 2, P0-3 covers this.

---

## 2. Changes (prioritised)

### P0-1. Replace the val-seed pseudo-test bank with leave-subject-out validation over pooled train+val subjects
- **Pool and fold.** Pool the 16 D1/D2 fall subjects and the 8 D3 subjects of train+val, plus the 8 ADL file ids (`_0,_4,_5,_6,_9,_10,_11,_12`).
- **Build each fold with the official generator.** Use `scripts/build_streams.py` logic with the `test` settings (`adl_uses=2`, `fade 0.2`): train on 3/4 of the subjects and ADL ids, and generate held-out streams from the other 1/4. Each fold holds out about 2 D1 subjects, 2 D2 subjects, 2 D3 subjects and 2 ADL ids, which mimics test.
- **Rotate P3_3** through the held-out D3 slot in at least one fold of every repeat.
- **Repeats and splitter.** Run 5 repeats with different subject partitions. This removes both the subject leakage and the ADL-file leakage (data team N5). Splitters such as `sklearn.model_selection.GroupKFold` / `StratifiedGroupKFold` work over subject groups.
- **What to report per candidate:**
  - the mean ranking F1 over folds;
  - the **per-subject ranking F1 distribution**: p10, worst subject, sd;
  - a **cluster (subject) bootstrap CI** of the paired difference against the incumbent.
- **Promotion rule.** Promote only if the paired subject-bootstrap 90 % CI excludes 0 and the worst-subject F1 does not fall.
- **The 5 official val streams** remain a one-shot holdout.
- **Expected gain:** none directly. It prevents selecting post-processing or offsets that fit 4 subjects. The val-subject spread above (0.873–0.894 across subject pairs) is as large as most proposed improvements *(estimate: avoids 0.01–0.03 of selection error)*.
- **Verify:** the prototype's LOSO mean should land below the 0.884 val score. If LOSO and val disagree by more than 0.03, investigate before trusting either.

### P0-2. A D3 hedge, with "held-out-P3_3 recall" as a gating metric
- **Features.** Add range-gated features per radar mat zone: R29171 bins 40–68, R29444 bins 5–21, R30722 bins 5–21. Each zone is normalised by its own radar's median, and is computed for:
  - static change at 1 s and 2 s;
  - frame-diff energy on distinct frames;
  - lag-1 coherence (X1, P1-1).

  These raise the D3 event-level AUC from 0.740 to 0.994.
- **Detection path for D3 (and as a second opinion for D1/D2): segment-level and mat-agnostic.**
  1. Junction detector → candidate short segment (7–19 s).
  2. Fall-clip classifier using cues that do not depend on the mat: spacing-3 repeat share (AUC 0.997/1.000), fingerprint, segment length, and the max zone-normalised static change.
  3. If the clip is fall-like, run X1's `propose_fall` replay, which picks the anchor radar by itself and therefore works on R29171.
- **Training.** Keep flagged D3 falls as positives in the frame model (see 1.3), but give the D3 path its own threshold.
- **Metrics.** Held-out-P3_3 recall @0.3/0.5 must be reported for every candidate. Also report the FP rate per hour on R29171-dominant clips: Autre/Empty and flagged D3 clips are free only within ±2 s.
- **Expected gain:** 0 on val, which has no evaluable D3. On test the expected gain is +0.02 to +0.03 with a tail of up to +0.08 to +0.10 *(estimate, 1.3)*.
- **Verify:** LOSO folds that hold out P3_3. The bar is recall ≥ 12/16 at θ0.3, with no drop on D1/D2 folds.

### P0-3. Decode at clip level, one fall per fall clip
- **Steps:**
  1. Segment the stream with the variance-normalised junction detector. Choose the threshold for fall-junction recall: q0.95–0.97 of train gives 0.94–0.95 (L2's table).
  2. Compute P(fall clip) for each segment.
  3. Emit **at most one** event per segment whose P(fall clip) is high, localised by X1's replay or selector, with the frame model as fallback.
  4. Never emit in segments over 60 s (Autre/Empty), except a D3-style replay hit when the anchor is R29171 and the static change is far above normal. This guards against a test generator that differs.
  5. Score = P(fall clip) × localisation confidence, so the greedy matcher sees good falls first.
- **Fallback.** Keep the frame-level detector for segments whose junction is uncertain, for example when two candidate junctions lie within 2 s.
- **Expected gain** *(estimate)*: the fragment and one-per-clip rule is worth about +0.004 (measured upper bound). The rest comes from turning the 4 mislocalised FP+FN pairs into hits, together with X1's replay, which measured 0.884 → 0.904–0.909 on val. Upside about +0.02–0.03.
- **Verify:** on LOSO, report:
  - the count of FP+FN pairs inside fall clips (target ≤ 1 per 70 falls);
  - the fragment count (target 0);
  - real FAs per hour outside fall clips.

### P0-4. Treat post-fall motion as an explicit negative
- **The failure mode.** Two of the 6 val FPs are post-fall motion on Chute04 clips. Post-fall movement and recovery attempts are a standard confounder in radar fall detection. The GT rule (`annotate._transition_bounds`) ties the fall to the **state rise** and caps expansion at +2 s, so motion after the settle is not part of the fall.
- **Features (per block, relative to the clip):**
  - time since the block where the anchor-zone `state` crosses 0.8;
  - "state already high" (person already on the floor);
  - motion energy relative to the clip's pre-fall motion.
- **Labels.** Label blocks more than 0.5 s after `t_end` in fall clips as a separate class: "post-fall", a 3-class LightGBM objective `multiclass`. Do not lump them with ADL.
- **Expected gain** *(estimate)*: removes about 2 FP+FN pairs on val-like data, worth roughly +0.01–0.02 ranking F1.
- **Verify:** on LOSO, report:
  - FPs whose start is after the GT end in the same clip (target 0);
  - Chute04 hit@0.5 per subject (val now 0.25 / 0.60 / 0.67 / 0.00).

### P1-1. Subject-invariant boundary offsets
- **Do not** fit a global start/end shift on val (data team: padding +0.01–0.02, not significant). The per-subject signed start bias runs from −0.13 to +0.22 s.
- Fit any residual offset inside LOSO, conditioned on fall type and the replay's internal signals, never on subject.
- **Metrics:** per-subject median |start error| and its spread, which is tie-break 1, and the share within ±0.1 s.

### P1-2. Robustness to unseen ADL subjects (the test ADL files are new)
- ADL ids seen so far are train 4, 5, 6, 9, 10, 12 and val 0, 11. Test ADL comes from other recordings *(inference)*.
- In LOSO with ADL ids held out, report the **per-activity score margin**: the max block probability per ADL chunk, by activity.
- Watch the movement-heavy activities (Faire_menage, Faire_vaisselle, Faire_pates, Faire_the), and Dormir (lying down, the classic confounder for static-change fall cues).
- **Promotion requires** that p99 of the per-chunk max probability on held-out ADL stays below the event threshold minus 0.1.
- Do **not** rely on the ADL duplicate mask for this. It is a generator artefact, pending the team lead's ruling.

### P1-3. Keep flagged falls as positives; do not cut them out of training
- Measured: dropping flagged fall blocks while keeping their neighbourhoods as negatives, combined with zone-agnostic features, gave val 0.797 against 0.884.
- To exclude junction-start and hard-cut falls (data-team N6), mask the **whole fall clip** from the loss, not just the event interval.
- **Verify:** an ablation on LOSO.

### P2-1. Treat the LIARA deep-model results as segment classifiers, not localisers
- The ResNet/ViT results (Abudalfa & Bouchard 2024/2025; about 95 % F1 under LOSO, per the preprint abstract) classify pre-cut windows.
- Here they fit stage S2 (P(fall clip) per segment) better than S3-c/d localisation. A torchvision `resnet18` on per-radar zone maps (|z| high-passed, coherence) over the whole segment, trained LOSO, is a reasonable diverse S2 member.
- Keep it P2: the cheap clip cues are already at AUC 0.99+.

### P2-2. Radar voting
- I agree with X1: "≥ 2 radar agents agree" contradicts the one-radar-per-mat geometry. Use anchor-radar logic.

### Plan text to amend
- **§2 step 2:** "pseudo-test bank from val segments" → "LOSO over pooled train+val subjects and ADL ids, generated with the test settings; subject-cluster bootstrap".
- **§5 risk "D3 low quality":** → "an evaluable D3 test subject is plausible (P3_3: 16 evaluable falls); held-out-P3_3 recall is a gating metric".
- **§3 S2/S4:** add "one event per fall clip" decoding (P0-3) and the post-fall class (P0-4).
- **§1 metrics:** add the items in §3 below.

## 3. Metrics to add
- Per-subject ranking F1 on LOSO: mean, p10, worst, and the subject-cluster bootstrap CI of the paired difference.
- Held-out-P3_3 recall @0.3/0.5, and recall per mat zone (D1/D2/D3).
- Count of FP+FN pairs in the same fall clip, fragments per clip, and post-fall FPs (FP start > GT end in the same clip).
- Real FAs per hour outside fall clips, split into: ADL by activity, Autre/Empty, and junction ±2 s.
- Per-activity max-score margin on held-out ADL ids (p99 against the threshold).
- Per-subject signed and absolute start-error median (tie-break 1), and per-type × subject hit@0.5.

## 4. Risks
- **Test subject mix.** About 2 subjects per mat *(inference)*, so the test score is a draw over very few people. Optimise the worst-subject score, not the mean on 4 val subjects.
- **D3 tail risk.** If a P3_3-like subject is in test, the current model loses about 16 falls (0/16 measured on held-out P3_3).
- **LOSO cost.** Regenerating streams per fold costs disk and time: about 400 MB per raw stream. Generate cached features directly from `segments/*.npy` crops, and do not write h5.
- **Small numbers.** Held-out-subject recall rests on 16–18 events per subject. The zone-feature results are single runs without retuned post-processing, and the variant that also dropped flagged blocks (P1-3) confounds two changes, so it is not a clean ablation.
- **Generator artefacts** (one-fall-per-clip, repeat spacing, ADL duplicates) may differ on test. Keep the physics-only frame detector as the hedge member, as the plan already says.

Sources: [Maitre, Bouchard & Gaboury, JBHI 2021](https://constellation.uqac.ca/6269/) · [IEEE Xplore 9212552](https://ieeexplore.ieee.org/abstract/document/9212552/references) · [Abudalfa & Bouchard, transfer-learning preprint](https://easychair.org/publications/preprint/hK22)

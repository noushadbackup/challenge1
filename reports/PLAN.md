# ABC 2027 UWB Fall Challenge — Plan to win

Status: **v1, before team review** (the 20-agent review below may amend it; the amended version
is kept in this file with a change log at the bottom).

Evidence base: `reports/REPORT.md`. Current verified score: **0.884** val ranking F1 (prototype,
official evaluator, `--ignore-flagged`) vs official baseline **0.315**.

---

## 0. What decides the ranking

`ranking = mean(F1@θ0.1, F1@θ0.3, F1@θ0.5)`, tie-break 1: lower median |start error| @θ0.3,
tie-break 2: fewer false alarms/hour @θ0.3. Predictions are matched greedily by **score**, so
the score column must rank true falls above doubtful ones.

The prototype is at ~0.95 for θ0.1/0.3 and **0.785 at θ0.5** → the remaining points are in
**boundary precision** (mean tIoU 0.70 now; target ≥ 0.80) and in the last few FPs/FNs.

## 1. Metrics we track (every experiment, every fold)

**Official** (from `abc2027.evaluate`): ranking F1; P/R/F1 at θ 0.1/0.3/0.5/0.7; FA/hour;
start/end error (median & mean |·|); fraction within ±0.5 s / ±1 s; mean tIoU; latency.

**Internal diagnostics**
| metric | why |
|---|---|
| frame-level ROC-AUC and PR-AUC (0.2 s blocks) | model quality before post-processing |
| recall per dataset (D1/D2/D3) and per fall type (Chute01–04) | find systematic misses |
| FP breakdown by context (junction ±1 s / ADL label / Autre / Empty / fall-clip margin) | target the dominant FP source |
| start/end error histograms (signed) | detect biased boundaries → offset correction |
| junction detector recall/precision (±0.5 s) | quality of the segment stage |
| corpus-classifier AUC (fall clip vs ADL vs Autre/Empty) | quality of the context stage |
| score calibration (reliability curve) | greedy matching uses the score order |
| **pseudo-test distribution**: ranking F1 mean ± sd over ≥ 50 regenerated val streams | variance-aware model selection (val has only 67 evaluable falls) |
| train-CV vs val gap | overfitting alarm |

## 2. Validation protocol (how every claim gets verified)

1. **Inner loop:** leave-streams-out GroupKFold on train (group = stream; also try group = fall subject).
   All hyper-parameters and post-processing are chosen here.
2. **Pseudo-test bank:** regenerate many streams from the *validation* segments in `segments/`
   with new seeds using the official generator (`scripts/build_streams.py`, test settings:
   `adl_uses=2`, flagged falls → `ignore`). Score each candidate on ≥ 50 such streams → mean ± sd.
   This mimics the hidden test (same generator, unseen seed, unseen subjects) far better than 5 fixed streams.
3. **Holdout:** the 5 official val streams are scored only for go/no-go decisions, never for tuning.
4. **Final fit:** retrain on train + val (+ regenerated streams), with settings frozen from 1–2.
5. **Promotion rule:** a change is kept only if it improves the pseudo-test mean by more than
   1 sd of the paired difference *and* does not lower holdout val.

## 3. Pipeline (stages and specific open-source models)

```
raw (3 × T × 368)
  └─ S0 preprocessing: I/Q → complex 184 bins; per-radar normalisation;
        junction-immune motion (frame diff, phase velocity); static profile at 0.2 s
  └─ S1 junction detector  ─┐
  └─ S2 segment context     ├─► S3 frame/segment fall model ─► S4 boundary refinement ─► S5 ensemble + calibration ─► CSV
        (length, fingerprint, static change)
```

| stage | model (all open-source) | input | output |
|---|---|---|---|
| S1 junctions | rule on static-profile jump + LightGBM refiner | 0.2 s profile jumps | junction times |
| S2 context | LightGBM / CatBoost | PCA fingerprint, segment length, static change | P(fall clip), P(Autre/Empty) |
| S3-a | **LightGBM / XGBoost frame classifier** (prototype, ≈0.88) | 0.2 s tabular features ± context lags | P(fall) per block |
| S3-b | **1D TCN / MS-TCN++** (PyTorch) | per-block feature sequence | framewise P(fall) + start/end heatmaps |
| S3-c | **ActionFormer** or **TriDet** (temporal action localisation; MIT-licensed repos) | per-block embeddings (from S3-b or S3-d encoder) | segments with regressed boundaries |
| S3-d | **ResNet-18 / EfficientNet-B0 (timm/torchvision)** on range-time images, 3 radars as channels (the cited LIARA approach) | 4 s windows of \|z\| and phase-velocity maps | window embedding / score |
| S3-e | **CRNN (DCASE sound-event-detection style)**: Conv over range, BiGRU over time | range-time sequence | framewise P(fall) |
| S3-f | **MiniRocket** (sktime) | per-radar block features | window score (cheap, diverse) |
| S4 boundaries | change-point on static profile (**ruptures** PELT/Binseg) + learned start/end offset regressor (LightGBM) | candidate ± 3 s | refined t_start, t_end |
| S5 ensemble | weighted average of framewise probs; **1D weighted segment fusion** (WBF analogue) + soft-NMS; isotonic calibration | all models | final events + score |

Optional open-source LLMs (e.g. Qwen2.5-Coder, Llama 3.x) are useful only as engineering
assistants (code review, report writing); they are not part of the detector.

### Multi-agent / voting, done where it helps
1. **Radar agents:** one model per radar + one fused model; a fall is accepted when the
   fused score or ≥ 2 radar agents agree (D1 is seen by R29444, D2 by R30722).
2. **Model committee:** S3-a…f vote at frame level (soft vote) and at event level (segment fusion);
   weights learned on the inner-loop OOF predictions (stacking with a logistic meta-model).
3. **Seed / fold bagging:** 5 folds × 3 seeds per model.
4. **Hedge member:** a physics-only model (no junction/fingerprint features) is kept in the
   ensemble so that the system still works if the test generator differs from the public one.

## 4. Phases

| phase | goal | exit criterion |
|---|---|---|
| P1 Infrastructure | feature cache, pseudo-test generator, metrics dashboard | pseudo-test bank of ≥ 50 streams; prototype re-scored on it |
| P2 Strong tabular | LightGBM with richer features (phase velocity, per-bin static change near mat bins, segment features), lag stacking | pseudo-test ranking F1 ≥ 0.90 |
| P3 Boundaries | change-point + offset regressor; score = calibrated P | F1@θ0.5 ≥ 0.88, median start error ≤ 0.2 s |
| P4 Deep models | TCN/CRNN, ResNet-18 encoder → ActionFormer | each ≥ 0.85 alone and adds ≥ +0.01 in ensemble |
| P5 Ensemble | stacking, segment fusion, radar voting | best pseudo-test mean; holdout val not worse |
| P6 Freeze & submit | retrain on train+val, reproducible package (code + weights + README) | runs end-to-end from raw .h5 in one command |

## 5. Risks and mitigations

| risk | mitigation |
|---|---|
| Test generator differs (structure features break) | physics-only hedge member; check ablation gap every phase |
| Over-tuning to 5 val streams | pseudo-test bank + promotion rule |
| Annotation bounds ±0.5–1 s noisy | regress to the annotation rule (static-profile change), not to motion peaks |
| D3 low quality | ignored on test → don't optimise for it; predictions there are free |
| Rule: no external LIARA data | only the package (incl. its `segments/`) is used; documented in the submission |
| Top-team code verification | everything scripted, seeded, versioned in this repo |

## 6. Team (20 agents)

| group | size | roles |
|---|---|---|
| Data team | 10 | 8 analysts each verify/challenge one finding of `REPORT.md` and vote (agree/disagree + evidence); 2 leads justify the votes and debate disputed items, then write the consolidated verdict |
| Field experts | 5 | UWB radar signal processing · temporal action localisation · fall-detection/eldercare sensing · ML validation & competition strategy · MLOps/reproducibility — each reviews this plan and proposes changes |
| Executors | 5 | E1 pseudo-test bank · E2 boundary refinement · E3 phase/Doppler features · E4 deep model (TCN/CRNN) · E5 ensemble & radar voting — each implements and verifies one work package |

## Change log
* v1 — initial plan from the analysis.

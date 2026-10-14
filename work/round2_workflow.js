export const meta = {
  name: 'uwb-fall-round2-win',
  description: 'Round 2 to win: build submission pipeline + subject-disjoint CV, evaluate, promote improvements, integrate, audit',
  phases: [
    { title: 'Build', detail: 'pipeline engineer (uwbfall package) + validation engineer (SR-CV harness), in parallel' },
    { title: 'Evaluate', detail: 'score current candidates on SR-CV, set the reference' },
    { title: 'Improve', detail: 'clip decoding, boundaries/D3, scoring model — each must pass the promotion rule' },
    { title: 'Integrate', detail: 'final config, final models on train+val, submission dry run' },
    { title: 'Audit', detail: 'reproducibility/compliance auditor + skeptical reviewer, then fixer' },
  ],
}

const CTX = `
CONTEXT — ABC 2027 UWB fall challenge. Our goal is to WIN. You are part of the round-2 engineering team.
- Task: in continuous UWB radar streams (3 radars x T x 368 @50 Hz; 368 = 184 complex range bins, I=0..183, Q=184..367) output one row per fall: stream_id,t_start,t_end,score[,t_alarm]. Ranking = mean event F1 at tIoU 0.1/0.3/0.5 (official evaluator, --ignore-flagged); tie-breaks: median |start error| @0.3, then false alarms/hour @0.3. Predictions are matched greedily by score.
- Package (READ-ONLY, never modify): /home/user/data_raw/abc2027_uwb_train_v1 (streams/{train,val}/stream_XX.h5, streams_events.csv, streams_composition.csv, segments/*.npy, src/abc2027/{evaluate,streams,annotate,align,baseline}.py, scripts/build_streams.py). Test streams will appear later as streams/test/stream_XX.h5 with ids test_00..test_04 (no labels).
  Official scoring: cd <package> && PYTHONPATH=src python3 -m abc2027.evaluate --gt streams_events.csv --pred <ABS csv path> --streams streams/val --ignore-flagged --out <json>
- Repo: /home/user/challenge1. MUST READ FIRST: reports/PLAN.md (v2 — the plan you are executing), reports/team/DATA_TEAM_VERDICT.md, and the READMEs of work/E1_pseudo_test (bit-exact generator/extractor/scorer replicas: e1lib.build_stream, e1lib.extract, scorer.py), work/E2_boundaries (annotator replay on distinct frames, frozen config B in out/summary.json), work/E4_deep_model (TCN; data.py builds 112-channel 0.2 s blocks), work/E5_ensemble, work/INTEGRATION (blend 0.75 TCN + 0.25 prototype HGB, then replay B = val 0.9646).
- Current best on val (official CLI): 0.9646 = E4 blend + E2 replay B (65/0/2 @0.3). But val has only 67 evaluable falls (noise ~±0.04) and stream-grouped train OOF is leaky, so nothing is trusted until it passes SR-CV.
- Promotion rule (PLAN v2 §2): a change is kept only if, on pooled subject-disjoint regenerated CV (SR-CV) OOF, a paired subject-cluster bootstrap gives P(delta ranking F1 > 0) >= 0.90 AND >= 3 of 4 folds improve AND official val does not drop by more than ~0.02. Seed-average (>=3 seeds) stochastic models before comparing. Val is for confirmation only, never for tuning.
- Rule compliance: only package data/code (incl. segments/ and generator) may be used; no external LIARA data; open-source libraries only; no pretrained weights needed. The cross-stream ADL duplicate mask (transductive across test streams) must stay OFF by default.
- Machine: 4 CPUs, 15 GB RAM, ~14 GB free disk, shared with one other agent at a time. Keep each process < 5 GB RSS, n_jobs/threads <= 2 (OMP_NUM_THREADS=2). NEVER store regenerated raw streams as h5 (disk): materialise cir in memory from composition rows (e1lib.build_stream) and cache only features (/home/user/data_raw/...). Put big caches/model files outside git or under paths matched by .gitignore (work/*/cache/, *.npz are ignored; add patterns to .gitignore if you create other big artefacts, e.g. models/*.pt over 5 MB).
- Installed: numpy scipy pandas scikit-learn lightgbm xgboost catboost torch(cpu) ruptures h5py matplotlib.
- Engineering rules: every number you report must come from code you ran in this session, with the official evaluator code. Do NOT git commit/push (the lead commits). Do not edit reports/REPORT.md or reports/PLAN.md. Write your own README/report as instructed. Be honest about limitations. If you hit something blocking, document it precisely and deliver the best verified partial result.`

const RESULT_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['done', 'partial', 'failed'] },
    summary: { type: 'string', description: 'what you built/found, key numbers, decisions' },
    results_table: { type: 'string', description: 'markdown table of key numbers' },
    interfaces: { type: 'string', description: 'exact commands / python APIs the next agents must use' },
    promoted: { type: 'array', items: { type: 'string' }, description: 'changes that passed the promotion rule and are now enabled by default (empty if none)' },
    files: { type: 'array', items: { type: 'string' } },
    open_issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'summary', 'results_table', 'interfaces', 'promoted', 'files', 'open_issues'],
}
const brief = r => r ? JSON.stringify({ status: r.status, summary: r.summary, results_table: r.results_table, interfaces: r.interfaces, promoted: r.promoted, open_issues: r.open_issues }) : 'AGENT FAILED (null result) — check its files on disk before relying on it.'

phase('Build')
const [pipe, srcv] = await parallel([
  () => agent(`You are the PIPELINE ENGINEER (PLAN v2 work package W6).
${CTX}

Build an installable, deterministic Python package /home/user/challenge1/uwbfall/ (importable with PYTHONPATH=/home/user/challenge1, plus a small pyproject or setup.cfg) that implements the CURRENT BEST SYSTEM end to end from raw streams to submission CSV:
  features (prototype 64 + E4 range bands = E4's 112 channels; port the code, do not import from work/), models (prototype HGB with early_stopping=False + TCN bag of 3 seeds, E4 hyper-parameters and stopping epoch), blend 0.75 TCN + 0.25 HGB, the prototype/E4 post-processing (operating point chosen on training OOF), and E2 replay config B for boundaries (median of 6 short crops on distinct frames, gate tIoU>=0.1, 1.0 s duration floor extending the end only).
Design requirements:
  1. Stream sources: (a) a directory of stream_XX.h5 files with an id prefix (train/val/test), (b) "virtual" streams given as composition rows rebuilt in memory with the E1 bit-exact writer (port e1lib.build_stream). The replay needs raw cir crops, so it must work for both.
  2. Python API: fit(train_sources, events_df, config) -> bundle saved to a directory; predict(bundle, sources) -> DataFrame. Everything driven by a JSON config with feature flags (so later agents can add options: e.g. decode.one_event_per_clip, score.hit_model, boundary.* ...). Fixed seeds, deterministic LightGBM/HGB settings, torch.manual_seed + torch.use_deterministic_algorithms where possible, single-thread or fixed-thread math for reproducibility.
  3. CLI: python -m uwbfall train --config <json> --train-splits train[,val] --out <model_dir>; python -m uwbfall predict --models <model_dir> --split val|test --out <csv> (runs the official validate_predictions before writing); python -m uwbfall score --pred <csv> --split val.
  4. Feature cache keyed by stream content hash + extractor version under /home/user/data_raw/uwbfall_cache/.
  5. RAM < 5 GB: process streams one at a time.
VERIFY: train on the 13 train streams only, predict val, score with the official CLI. Target: reproduce the integration result 0.9646 (65/0/2) — exact match is ideal; if your number differs, find out why (seed/order/threads) and either fix it or explain it with evidence. Also report wall time of train and of predict (predict must handle 5 test streams in reasonable time).
Write /home/user/challenge1/uwbfall/README.md (architecture, config schema, commands, verification results). Time-box ~2 hours.`,
    { label: 'Pipeline engineer (uwbfall)', phase: 'Build', schema: RESULT_SCHEMA }),
  () => agent(`You are the VALIDATION ENGINEER (PLAN v2 work package W1, SR-CV).
${CTX}

RESUME NOTE: a previous attempt at this exact task was cut off by a usage limit. It left partial code in /home/user/challenge1/srcv/ (README.md, build.py, bridge.py, baseline_prototype.py, cache_feats.py, compare.py, configs/) and possibly data under /home/user/data_raw/srcv/. Review it critically, reuse what you can verify, fix what is wrong, and finish. The pipeline package now EXISTS at /home/user/challenge1/uwbfall (read uwbfall/README.md): its sources accept "comp:<composition.csv>[:<events.csv>][@<prefix>]" and VirtualSource(sid, segments) — make SR-CV sources directly consumable by uwbfall.fit/predict.

Build /home/user/challenge1/srcv/ — subject-disjoint regenerated cross-validation shaped like the official test:
  1. Pool train+val fall subjects (Test_P{dataset}_{n}, with their own Chute*/Autre*/Empty* recordings) and ADL subjects (trailing number of activities/<Act>_<n>.json; there are 8 ADL subjects). Read the package generator code (src/abc2027/streams.py, scripts/build_streams.py _groups) to learn how fall rows (t_start,t_end,t_last,quality,needs_review,fall_type,dataset) are built; reconstruct them from streams_events.csv (source_t_start/end, quality, needs_review) and segments/*.npy shapes (t_last), as E1 did for val. Verify your reconstruction against the published events for val and train.
  2. 4 folds: each held-out fold = ~2 D1 + 2 D2 + 2 D3 fall subjects (balance so every D3 subject incl. Test_P3_3, the only mostly-evaluable D3 subject, is held out once) + 2 ADL subjects. Save folds.json.
  3. For each fold k: TRAIN streams regenerated from the remaining subjects with split='train' settings (adl_uses=3), TEST streams from the held-out subjects with split='test' settings (adl_uses=2, flagged falls -> ignore=true), using the official prepare_split/finish_plan with fixed seeds and density 57.1864, plus the E1 bit-exact writer. Store ONLY composition + events CSVs (no raw h5) under /home/user/data_raw/srcv/fold_k/; the pipeline materialises cir on demand.
  4. Scoring: score(pred_df, fold) with official abc2027.evaluate code and durations from the composition; pooled OOF scoring across folds; promote(pred_A_by_fold, pred_B_by_fold) implementing the promotion rule (paired subject-cluster bootstrap: attribute each FP to the segment/subject it falls in, 5000 resamples; P(delta>0), per-fold deltas, n folds improved).
  5. A ready-made baseline run: score the prototype recipe (analysis/prototype_detector.py 'full' features via E1's extractor replica, HGB, post-processing chosen on the fold's training OOF) on all 4 folds so the harness is proven end to end before the pipeline package exists.
Provide a clean Python API (srcv.folds(), srcv.sources(fold, part) returning composition-based stream specs usable by a pipeline, srcv.events(fold, part), srcv.score(...), srcv.promote(...)) and document it in /home/user/challenge1/srcv/README.md, coordinating with the pipeline engineer's planned interface: the pipeline package will be at /home/user/challenge1/uwbfall with fit(train_sources, events_df, config)/predict(bundle, sources) and "virtual stream" sources built from composition rows — make your sources directly consumable by it (document the exact dict/dataclass format). Time-box ~2 hours.`,
    { label: 'Validation engineer (SR-CV)', phase: 'Build', schema: RESULT_SCHEMA }),
])
log(`Build: pipeline=${pipe ? pipe.status : 'null'}, srcv=${srcv ? srcv.status : 'null'}`)

phase('Evaluate')
const evalRes = await agent(`You are the EVALUATION ENGINEER.
${CTX}

Results of the build phase:
PIPELINE (uwbfall): ${brief(pipe)}
SR-CV harness: ${brief(srcv)}

First make the two fit together (fix small interface mismatches in either package if needed, documenting what you changed). Then score these candidates on all 4 SR-CV folds (train on the fold's training streams, predict its test streams; same seeds) and on official val (trained on the 13 train streams):
  C0 prototype HGB alone (detector bounds); C1 TCN 3-seed bag alone; C2 blend 0.75 TCN + 0.25 HGB; C3 = C2 + replay B; C4 = C1 + replay B.
Report a table: pooled SR-CV ranking F1 (mean and per fold), F1@0.1/0.3/0.5/0.7, TP/FP/FN, FA/h, median start error, and val. Run the promotion rule for C3 vs C2, C2 vs C1, C4 vs C1, C3 vs C0. Investigate the earlier puzzle (replay helped the blend on val but slightly hurt the bare TCN): find the mechanism with per-event evidence and propose a fix if one is obvious (e.g. gate on detector-interval width or replay agreement) — test it on SR-CV under the promotion rule.
Set the REFERENCE SYSTEM = the best candidate that passes the rules (default C3 if it passes) and make it the default config of uwbfall (configs/reference.json). Write /home/user/challenge1/reports/round2/EVAL.md. Keep total compute within ~2.5 hours (if TCN training per fold is too slow, use the E4 stopping epoch with fewer steps but keep 3 seeds, and say so).`,
  { label: 'Evaluation engineer', phase: 'Evaluate', schema: RESULT_SCHEMA })

phase('Improve')
const IMPROVE = [
  { id: 'I1', name: 'Clip-level decoding + segment classifier + post-fall negative class', task: `Implement (as config flags in uwbfall) PLAN v2 W3 and expert P0s X2/X3: (a) variance-normalised junction detector with threshold set for fall-junction recall (~q0.95); (b) LightGBM segment classifier fall-clip / ADL / Autre-Empty using static change, motion burst, fingerprint, spacing-3 repeat share (within-stream only) and segment length; (c) decode at most ONE event per junction-to-junction segment (hard 1-D NMS), gate frame posteriors by P(fall clip); (d) optionally a post-fall negative class (3-class GBDT: fall / post-fall-in-clip / other) feeding the blend. Test each sub-change separately and the best combination on SR-CV against the REFERENCE under the promotion rule. Enable by default only what passes.` },
  { id: 'I2', name: 'Boundaries: Chute04, D3 hedge, flagged-event loss masking', task: `Improve boundaries and robustness (PLAN v2 W4, experts X1/X3): (a) Chute04 (chair falls) boundary errors — analyse replay failure modes per fall type on SR-CV OOF and fix (e.g. crop choice per detected context, replay residual LightGBM regressor on replay internals/crop spread, anchor-radar choice); (b) D3 hedge: zone features for R29171 bins 40-68 (and 5-21 for the other radars), measure recall on the evaluable D3 subject Test_P3_3 in its SR-CV fold; (c) training labels: keep flagged falls as positives but give zero loss weight to whole clips that are hard-cut/junction-start ignore zones. Never tune a start pad (protect tie-break 1). Each sub-change tested under the promotion rule against the current default config; enable only what passes.` },
  { id: 'I3', name: 'Score model and operating point', task: `Make the score column and the decision threshold optimal for the ranking metric (PLAN v2 W5, expert X4): (a) event-level hit model (LightGBM; targets hit@0.1/0.3/0.5) on SR-CV OOF candidate events with features: detector peak/mean probability, replay-vs-detector tIoU, multi-crop spread, segment-class probability (if available), duration, junction distance, radar anchor strength; score = expected mean hit; (b) choose the operating threshold that maximises pooled mean F1 over theta .1/.3/.5 with nested CV (no leakage), check stability across folds; (c) verify that predictions overlapping ignore zones are never suppressed (they are free). Test under the promotion rule against the current default config; enable only what passes.` },
]
let improveResults = []
let state = brief(evalRes)
for (const imp of IMPROVE) {
  const r = await agent(`You are improvement engineer ${imp.id}: ${imp.name}.
${CTX}

Evaluation/reference state so far (latest first is most relevant):
${state}

Previous improvement results this round: ${improveResults.map(x => brief(x.r)).join('\n') || 'none'}

YOUR TASK: ${imp.task}
Implement inside the uwbfall package behind config flags (keep the default config unchanged unless your change passes the promotion rule; if it passes, update configs/reference.json and say so). Report SR-CV pooled + per-fold numbers and val for each variant, with the promotion-rule outcome. Write /home/user/challenge1/reports/round2/${imp.id}.md. Time-box ~2 hours of compute.`,
    { label: `${imp.id} ${imp.name}`, phase: 'Improve', schema: RESULT_SCHEMA })
  improveResults.push({ id: imp.id, r })
  if (r) state = brief(r) + '\n' + state
}

phase('Integrate')
const integ = await agent(`You are the INTEGRATOR.
${CTX}

Evaluation: ${brief(evalRes)}
Improvements: ${improveResults.map(x => x.id + ': ' + brief(x.r)).join('\n')}

1. Freeze the final config (configs/final.json) = reference + every PROMOTED change. Re-run it on all 4 SR-CV folds and val in one clean run and report the final table (with promotion-rule comparison vs the round-1 integration C3).
2. Train the FINAL models on train + val (all 18 streams; operating point from the CV OOF, not from val) into /home/user/data_raw/uwbfall_models/final/ (copy small artefacts + a manifest with sha256 into /home/user/challenge1/models/final/ if under ~20 MB total, else document location and how to regenerate).
3. Submission dry run: treat the 5 val streams as if they were test (stream ids test_00..04 via a temporary renamed copy in /home/user/data_raw/fake_test/ — do not modify the package) and run the exact test command end to end: python -m uwbfall predict --models ... --split test ... ; make sure validate_predictions passes and timing is acceptable. Also add label-free drift monitors (predicted events/h, detected junctions/h, share of predictions near junctions, per-radar feature PSI vs training) printed by predict, plus a physics-only fallback config.
4. Write /home/user/challenge1/SUBMISSION.md: exact steps on release day (download test, one command, checks, fallback), and /home/user/challenge1/Makefile targets (features, train-final, predict-test, reproduce-val, srcv). Time-box ~2 hours.`,
  { label: 'Integrator', phase: 'Integrate', schema: RESULT_SCHEMA })

phase('Audit')
const [audit, review] = await parallel([
  () => agent(`You are the REPRODUCIBILITY & COMPLIANCE AUDITOR.
${CTX}

Integrator result: ${brief(integ)}

Audit like a challenge organiser verifying a top team: (1) fresh clone of the repo into /home/user/data_raw/audit_clone (git clone /home/user/challenge1 — note uncommitted files: copy the working tree with rsync excluding big caches if needed), install per README, run the documented commands, check that reproduce-val gives the reported val CSV bit-exactly (or document every difference); (2) run predict twice and with different thread counts — determinism; (3) validate_predictions on the dry-run CSV; (4) rule compliance: grep for any use of external data, transductive cross-stream tricks enabled by default, val/test labels used in training or tuning, package files modified (compare package manifest sha256 fb0e2402... recorded in work/E1_pseudo_test/manifest_before.txt); (5) RAM/time of predict on 5 streams. Write /home/user/challenge1/reports/round2/AUDIT.md with PASS/FAIL per item and exact fixes needed.`,
    { label: 'Reproducibility & compliance auditor', phase: 'Audit', schema: RESULT_SCHEMA }),
  () => agent(`You are the SKEPTICAL REVIEWER (independent; your job is to find what is wrong before the leaderboard does).
${CTX}

Evaluation: ${brief(evalRes)}
Improvements: ${improveResults.map(x => x.id + ': ' + brief(x.r)).join('\n')}
Integrator: ${brief(integ)}

Try to refute the claimed final score: look for leakage (fold construction, ADL file reuse across fold train/test, subjects, operating point chosen on test folds, val used for tuning), metric mistakes (ignore zones, durations, theta), overfitting to generator artefacts that might not hold on the official test (e.g. sample-and-hold repeats, crossfade signature, seed-specific structure), and fragile thresholds. Re-run key numbers yourself where cheap. Estimate the expected official test score with an honest interval and list the top 3 risks with concrete mitigations. Write /home/user/challenge1/reports/round2/REVIEW.md.`,
    { label: 'Skeptical reviewer', phase: 'Audit', schema: RESULT_SCHEMA }),
])

const blocking = [audit, review].filter(Boolean).flatMap(r => r.open_issues || [])
let fix = null
if (blocking.length) {
  fix = await agent(`You are the FIXER.
${CTX}

Auditor: ${brief(audit)}
Reviewer: ${brief(review)}
Integrator: ${brief(integ)}

Fix every issue that blocks a valid, reproducible, rule-compliant submission or that the reviewer showed is a real leakage/metric error. For each fix: change the code, re-run the affected verification (reproduce-val, determinism, SR-CV numbers if the model changed), and record before/after numbers. Do not start new modelling work. Update SUBMISSION.md if commands changed. Write /home/user/challenge1/reports/round2/FIXES.md.`,
    { label: 'Fixer', phase: 'Audit', schema: RESULT_SCHEMA })
}

return { build: { pipeline: pipe, srcv }, evaluation: evalRes, improvements: improveResults, integration: integ, audit, review, fix }

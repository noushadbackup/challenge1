# R3 "boundary": sharper fall start/end boundaries

Base system: `final_norep`, SR-CV TCN bag 0. Its pooled SR-CV rank F1 is **0.9139**, with F1 @0.1/0.3/0.5/0.7 of 0.940/0.928/0.874/0.715 and a median start error of 0.04 s. Official val rank F1 is **0.9373**.

All numbers come from the official evaluator through `srcv`. Promotion uses `srcv.promote` against the base fold/val CSVs: a paired subject-cluster bootstrap with 5000 draws, ≥ 3/4 folds improved, and val ≥ base − 0.02.

## TL;DR

* **Diagnosis:** Errors at tIoU 0.5 come almost entirely from candidates whose replay crops *agreed* (passed the gate), not from gate-failed ones.
  * In the training OOF of all 5 targets, 52 of 64 tIoU-0.5 failures among detected falls are agreeing crops; 12 are gate-failed.
  * On the test streams it is 18 vs 4.
  * In 33 of the 52 agreeing training failures, the raw detector bounds were already ≥ 0.5.
  * Most agreeing candidates have all 6 crops agreeing (164/181 on fold 0). Crop-count adaptive rules therefore have nothing to work with.
* **Promoted: `wdet_cv`.** It blends the replay-median bounds with the detector (TCN/HGB decode) bounds on agreeing candidates:
  * start = (1−ws)·replay + ws·detector, and the same for the end with we.
  * (ws, we) ∈ {0, …, 0.5}² is chosen per target on its training OOF only.
  * Results: pooled SR-CV **0.9222 (Δ +0.0083, CI [+0.0024, +0.0157], P = 0.995, 3/4 folds; fold 3 is tied)**, val 0.9373 → **0.9424**. It passes the rule.
* **Robustness over TCN bags.** The same procedure was re-run on composite `final_norep` bundles built from TCN bags 1 and 2, each retuned on training OOF and compared with its own base:
  * bag 2 passes (Δ +0.0071, P 0.980, 3/4).
  * bag 1 does **not** (Δ +0.0018, P 0.662, 2/4).
  * The 3-bag seed-averaged promotion passes: **Δ +0.0057 [+0.0007, +0.0118], P = 0.989, 3/4 folds, val (3-bag mean) 0.9443 → 0.9493**.
  * Val improves by +0.005 in each of the 3 bags.
* **Requested dense boundary model (TriDet-style distribution head, soft labels): negative.**
  * On its own it is worse than the replay at tIoU 0.5: 0.857 vs 0.930 of matched falls in training OOF. It is also worse than the detector bounds (0.897).
  * As a fallback for gate-failed candidates it ties the detector at 0.5 (0.600 vs 0.600, n = 30).
  * As a blend member it adds nothing beyond `wdet_cv`: `model_cv` scores 0.9209 vs 0.9222.
  * As a fallback on top of `wdet_cv` (`wdet_fb`) it improves only 2/4 folds and fails the rule, although val goes up to 0.9574.
* **Stacked Huber regressors of the boundary offsets: negative.**
  * `stack_det`: P 0.876, 2/4 folds.
  * `stack_full`: Δ −0.0047.
  * Offset regression optimises typical error, while rank F1 is driven by the tail cases.
* **Caveat (tie-break):** The blend moves agreeing starts towards the (earlier) detector start. The median absolute start error at θ 0.3 (official tie-break 1) rises from 0.04 s to about 0.16–0.20 s, and mean tIoU drops from 0.824 to 0.782. F1 @0.7 changes by −0.003 (not ranked). The primary ranking key (mean F1 @0.1/0.3/0.5) improves.

## 1. Where do tIoU-0.5 errors come from? (`diag.py`)

For each evaluable GT fall we take the candidate with the largest detector-bounds tIoU (≥ 0.1 = detected) and its final replay bounds. Then we ask:
- whether any crop passed the gate ("agree") or not ("gate_fail", which falls back to detector bounds);
- what the oracle tIoUs are (best crop, best of crops + detector).

| (all 5 targets) | evaluable | det. miss | agree n | agree fail@0.5 | of which det ≥ 0.5 | of which oracle < 0.5 | gate-fail n | gate-fail fail@0.5 | agree fail@0.7 | gate-fail fail@0.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| training OOF (`diag_train.json`) | 945 | 29 | 886 | 52 | 33 | 13 | 30 | 12 | 180 | 26 |
| test (diagnostic only, `diag_test.json`) | 320 | 12 | 296 | 18 | 8 | 9 | 12 | 4 | 62 | 9 |

Per-event rows are in `diag_*_events.csv`. Two conclusions follow:
- A better fallback can fix at most a handful of falls.
- The replay and the detector make partly independent errors on agreeing candidates. In the training OOF at tIoU 0.5, the detector alone scores 0.907, the replay 0.941, and the best of the 6 crops plus the detector (oracle) 0.985.

## 2. Methods

All choices are made on the target's **training OOF** only, meaning each base bundle's own `oof_candidates.csv` replayed on its training streams. "Targets" are SR-CV folds 0–3 (bundles `runs/strict/norep/fold_k`) and val (`official_train`). No member is retrained.

* `dump.py`: replays the 6 crops for (a) the training-OOF candidates and (b) the test candidates. It writes `D_train.csv`, `D_test.csv` and `p_test.npz`. Check: `refine` + NMS reproduces each bundle's `oof_final.csv` and the base fold/val CSVs exactly (`logs/dump*.log`).
* **`wdet_*` (promoted family)** (`select_b.bounds`, `grid2.py`, `apply.py`):
  * Gate-failed candidates keep the detector bounds.
  * Agreeing candidates get `s = (1-ws)·s_rep + ws·s_det`, `e = (1-we)·e_rep + we·e_det`, then `e ≥ s + dmin`, then the unchanged NMS.
  * `grid2.py none` scores ws, we ∈ {0, .1, …, .5} on training OOF.
  * `wdet_cv` takes the argmax (ties → smaller ws+we, then smaller ws). `wdet_sym` restricts the search to ws = we.
  * Chosen (ws, we): fold0 (0.4, 0.1), fold1 (0.5, 0.4), fold2 (0.4, 0.2), fold3 (0.3, 0.0), val (0.5, 0.3).
  * `fixed_0.2_0.2` and `fixed_0.4_0.4` are sensitivity checks only. They were picked after looking at all targets' training OOF, so they are not eligible.
* **Dense boundary model `b3`** (`bmodel.py`, `bfit.py`):
  * Architecture: dilated residual 1-D CNN (h = 64, dilations 1–32) on the 0.2 s blocks.
  * Inputs (173 channels):
    * the 112 TCN channels of the bundle, built with the pipeline's `features.stream_blocks`/`design`, the bundle PCA and train-set standardisation;
    * 30 fine channels: log per-radar motion energy at the 10 frames of each block;
    * 30 frame-difference-energy channels (3-frame running max);
    * the detector probability (training OOF on training streams, final members on test streams).
  * Outputs: start and end logits on a 0.02 s grid.
  * Training (TriDet-style distribution head): soft-label cross-entropy (Gaussian σ = 0.06 s) of softmax(logits) over a ±4 s window around the GT boundary. The window is randomly shifted by ±2 s to simulate detector error. 800 steps, AdamW, one-cycle schedule, all falls of the training streams.
  * Training-OOF logits come from a 4-fold inner stream CV inside each target's training streams (bundle meta folds). The final model is trained on all training streams and applied to the test streams.
  * How it is used:
    * `fb_model`: the model's argmax bounds for gate-failed candidates.
    * `model_only`: blend of λ·model with the replay.
    * `model_cv`: ws, we and λ chosen jointly on training OOF.
    * `wdet_fb`: the `wdet_cv` blend plus a fallback (detector / model-argmax / model-expectation) chosen per target on training OOF.
  * `fbsel.py` implements `wdet_fb`. `mq.py` measures model quality on training OOF.
  * Earlier exploration on fold 0 only, kept in `sel_train_b1.json`: `b1` (143 channels, unflagged falls, 400 steps) and `b2` (143 channels, all falls, 800 steps). Neither was better.
* **Stacked regressors** (`stack.py`):
  * Model: HuberRegressor (α = 1) on standardised features that predicts GT − replay offsets, fitted on matched training-OOF candidates.
  * `det` uses only the detector−replay offsets. `full` adds the 6 per-crop offsets, crop mask, spread, low-quality flag and durations.
  * Its training-OOF score comes from an inner stream CV.
* **Bag robustness** (`bags.py`, `bag3.py`):
  * Composite bundles use the norep HGB plus the TCN members of `srcv/runs/tcn_bag{1,2}`, retuned with `uwbfall.pipeline.retune` on training OOF only.
  * They are then dumped (`R3_BAG=bagN dump.py`), gridded (`R3_BAG=bagN grid2.py none`) and applied (`R3_BAG=bagN apply.py wdet_cv`).
  * Each bag is compared with its own retuned base (`dump_bagN/*/base_test.csv`).

## 3. Results (SR-CV pooled over 4 folds, 253 evaluable falls; val = official val)

| variant | pooled rank F1 | base | Δ [95% CI] | P(Δ>0) | folds improved (per-fold Δ) | val base → cand. | F1 @.1/.3/.5/.7 | med. start err @.3 | rule |
|---|---|---|---|---|---|---|---|---|---|
| **wdet_cv** | **0.9222** | 0.9139 | +0.0083 [+0.0024, +0.0157] | 0.995 | 3/4 (+0.0160, +0.0114, +0.0054, +0.0000) | 0.9373 → 0.9424 | 0.946/0.934/0.887/0.712 | 0.195 | **yes** |
| wdet_sym | 0.9222 | 0.9139 | +0.0083 [+0.0024, +0.0157] | 0.995 | 3/4 (+0.0160, +0.0114, +0.0054, +0.0000) | 0.9373 → 0.9424 | 0.946/0.934/0.887/0.708 | 0.176 | yes |
| model_cv (wdet + dense model) | 0.9209 | 0.9139 | +0.0070 [+0.0017, +0.0138] | 0.995 | 3/4 (+0.0160, +0.0065, +0.0054, +0.0000) | 0.9373 → 0.9424 | 0.946/0.934/0.883/0.716 | 0.185 | yes, but < wdet_cv |
| model_only | 0.9152 | 0.9139 | +0.0013 [+0.0000, +0.0043] | 0.641 | 1/4 | 0.9373 → 0.9424 | 0.940/0.928/0.878/0.718 | 0.056 | no |
| fb_model | 0.9165 | 0.9139 | +0.0026 [−0.0026, +0.0097] | 0.726 | 1/4 (+0.0160, 0, −0.0054, 0) | 0.9373 → 0.9524 | 0.940/0.932/0.878/0.722 | 0.040 | no |
| wdet_fb | 0.9209 | 0.9139 | +0.0070 [+0.0000, +0.0150] | 0.970 | 2/4 (+0.0160, +0.0114, 0, 0) | 0.9373 → 0.9574 | 0.946/0.934/0.883/0.716 | 0.192 | no |
| stack_det | 0.9165 | 0.9139 | +0.0026 [+0.0000, +0.0063] | 0.876 | 2/4 | 0.9373 → 0.9424 | 0.940/0.932/0.878/0.738 | 0.045 | no |
| stack_full | 0.9092 | 0.9139 | −0.0047 [−0.0157, +0.0046] | 0.184 | 1/4 (0, +0.0114, −0.0269, −0.0051) | 0.9373 → 0.9273 | 0.938/0.926/0.864/0.696 | 0.069 | no |
| fixed_0.2_0.2 (sensitivity) | 0.9191 | 0.9139 | +0.0052 [+0.0012, +0.0099] | 0.987 | 3/4 | 0.9373 → 0.9424 | 0.940/0.932/0.885/0.715 | 0.116 | (not eligible) |
| fixed_0.4_0.4 (sensitivity) | 0.9222 | 0.9139 | +0.0083 [+0.0024, +0.0157] | 0.995 | 3/4 | 0.9373 → 0.9424 | 0.946/0.934/0.887/0.712 | 0.190 | (not eligible) |
| wdet_cv, TCN bag 1 (vs bag-1 base) | 0.9284 | 0.9266 | +0.0018 [−0.0060, +0.0098] | 0.662 | 2/4 (+0.0106, −0.0033, +0.0054, −0.0051) | 0.9502 → 0.9552 | 0.949/0.934/0.902/0.730 | 0.160 | no |
| wdet_cv, TCN bag 2 (vs bag-2 base) | 0.9218 | 0.9147 | +0.0071 [+0.0005, +0.0137] | 0.980 | 3/4 (+0.0106, +0.0070, +0.0107, 0) | 0.9453 → 0.9502 | 0.947/0.931/0.888/0.698 | 0.163 | yes |
| **wdet_cv, 3-bag seed-averaged** | 0.9241 | 0.9184 | +0.0057 [+0.0007, +0.0118] | 0.989 | 3/4 (+0.0124, +0.0050, +0.0071, −0.0017) | 0.9443 → 0.9493 | | | **yes** |

Notes:
- The base pooled F1 is 0.940/0.928/0.874/0.715, with TP/FP/FN @0.3 = 239/23/14 and start error 0.04 s.
- `wdet_cv` gives 240/21/13. The blend also changes some NMS and matching outcomes at θ 0.1 and 0.3, not only at θ 0.5.
- Per fold, the changes amount to 1–2 events, which is the usual SR-CV granularity.
- The training OOF is optimistic: about +0.01 per target after an argmax over 36 cells. The realised gain is +0.002 to +0.008 depending on the TCN bag.

Files: `results/<variant>/{fold_0..3,val}.csv` + `summary.json` (chosen parameters, training-OOF scores, per-fold scores, promotion output). The 3-bag output is in `results/wdet_cv_3bag.json`.

## 4. Assessment / recommendation

* `wdet_cv` is a zero-cost post-processing change. It touches only the boundaries of agreeing candidates and needs no new model.
* It meets the promotion rule on bag 0 and on the 3-bag average. Its gain is small (+0.006 averaged over 3 bags) and not significant on bag 1.
* **Deployment:** (ws, we) would be chosen on the final bundle's own training OOF with the same grid and tie-break. Every target chose ws ∈ [0.3, 0.5] and we ∈ [0, 0.4]; `wdet_sym` and fixed (0.4, 0.4) give identical SR-CV.
* **Cost:** a worse tie-break (median start error about 0.19 s instead of 0.04 s). If the organisers' primary key ties, this would lose.
* **Dense boundary head:** with about 190 training falls per target, it does not learn boundaries as precisely as re-running the package annotator on crops. Its errors are larger and only weakly complementary. It is not recommended.

## 5. Compliance

- Each stream is processed on its own.
- The blend and the dense model use only that stream's detector output, replay crops and block features.
- No cross-stream or within-stream matching or alignment, and no frame-repeat statistics:
  - The fine channels are motion energy and frame-difference energy, the latter smoothed by a 3-frame running max.
  - No repeat counting or detection is used.
- Only challenge data is used.
- Every parameter (ws, we, λ, fallback choice, regressor fit, dense-model training) uses the target's training streams only (training OOF via the bundles' inner folds). Test streams and val were used only for the final scoring.

## 6. Reproduce

```bash
export PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src ABC_ROOT=/home/user/data_raw/abc2027_uwb_train_v1 \
       PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd /home/user/challenge1/work/R3_boundary
python3 dump.py 0 1 2 3 val                      # replay internals -> /home/user/data_raw/r3/boundary/dump (~10 min)
python3 diag.py train && python3 diag.py test    # section 1
python3 grid2.py none                            # training-OOF grid (ws, we)
python3 apply.py wdet_cv wdet_sym fixed:0.2:0.2 fixed:0.4:0.4
for t in 0 1 2 3 val; do python3 bfit.py $t --tag b3 --steps 800 --all-falls --fine2; done   # dense model (~8 min/target)
python3 grid2.py b3 && python3 mq.py b3 && python3 apply.py model_cv model_only fb_model && python3 fbsel.py
python3 stack.py det full
for B in bag1 bag2; do python3 bags.py $B && R3_BAG=$B python3 dump.py 0 1 2 3 val && \
    R3_BAG=$B python3 grid2.py none && R3_BAG=$B python3 apply.py wdet_cv; done
python3 bag3.py
```

Intermediates are in `/home/user/data_raw/r3/boundary/` (97 MB):
- `dump*/`: replay tables;
- `logits/b{1,2,3}/`: dense-model logits and weights;
- `bags/bag{1,2}/`: retuned composite bundles.

The input cache `x/` and the composite source dirs were deleted, and the scripts rebuild them. Also in this folder:
- `grid2_none_tau.json` and `grid2_none_os.json`: extra training-OOF grids (conditional blend only when replay and detector disagree, and a one-sided start blend). Neither beat the plain blend.
- `shiftgrid.json`: a constant-shift ablation that recovers part of the gain.
- `oracle_clip.py`: a training-only diagnostic of annotator replay on true clips.

Resume note: the first attempt was cut off by a usage limit after `dump`, `diag`, the bag-0 `wdet` results and the fold-0 dense-model screen. These outputs were re-validated: the dump logs reproduce `oof_final` and the base CSVs, and the summaries were re-read. The later steps were then completed: dense model on all targets, bag 1/2 robustness, `fbsel` and `stack`.

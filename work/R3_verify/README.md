# R3 "verify": an image-based second-stage verifier on range-time crops (ResNet-18, ImageNet weights)

**Verdict: clean negative. Nothing is promoted.** Every variant fails the fold clause: only 1 of 4 folds improves, against 3 required. P(Δ>0) is 0.873, also below the 0.90 bar.

The verifier does find a real pattern in fold 1. That fold's held-out ADL subject produces "Laver_dents" (brushing teeth) false alarms, and the verifier rejects all 7 of them without losing a true positive. In the other three folds the training-OOF tuning keeps (almost) every base row, so those folds stay unchanged. Val stays identical to the base (0.9373, 63/3/4).

## What

- **Base system:** `final_norep`, bag 0. Its predictions are in `work/STRICT/results/norep/`. Its bundles are in `/home/user/data_raw/srcv/runs/strict/norep/`.
- **The new view:** each candidate is rendered as an image straight from the raw CIR, one stream at a time.
  - Image size: 3 channels (the 3 radars) × 72 range bins (0..71) × 80 time bins of 0.1 s (candidate centre ± 4 s).
  - Pixel value: log10 of the mean frame-difference energy |z_t − z_{t−1}|², where z = I + jQ.
  - Each value has the stream's own per-range-bin median subtracted. No information crosses streams, and no frame-repeat statistics are used.
- **The model:** ResNet-18 with torchvision's ImageNet weights.
  - Weights file: `resnet18-f37072fd.pth`, sha256 prefix f37072fd, downloaded from download.pytorch.org.
  - torchvision is not installed, so the architecture is re-implemented in plain torch in `cnn.py` and loads that state_dict unchanged.
  - The input is upsampled to 96×112.
  - Fine-tuning: AdamW (backbone lr 3e-4, head 1e-3), one-cycle schedule, 6 epochs, batch 32, BCE with positive-class weight.
  - Augmentation: ±0.5 s centre jitter, per-radar gain jitter and noise.
  - Runs on CPU with 1 thread, about 20 s per epoch.
- **Target:** hit = tIoU ≥ 0.3 with any fall event of the stream. Flagged falls count, since they are real falls.

### Training data (training streams of each fold only)

Built by `prep.py`. For each set (fold0..3; official = the 13 train streams → val) there are three kinds of rows:

| src | what | count per fold | hits |
|---|---|---|---|
| final | the bundle's own OOF rows after replay + NMS (`oof_final.csv`): the rows the verifier has to judge | 336–355 | 317–340 |
| low | extra detector candidates decoded from the bundle's OOF blend (`oof.npz`) at threshold 0.05 (smooth 1, min_dur 0.2, gap 0.2), excluding those overlapping a final row → hard negatives | 219–229 | 12–35 |
| rand | 25 random 1.5 s windows per training stream, ≥ 3 s from any fall → easy negatives | 325 | 0 |

### Choosing the cut (training data only)

`cnn.py <set> inner` produces out-of-fold verifier probabilities for the training candidates. It uses a subject-grouped 4-fold split, where the subject is that of the composition segment the candidate overlaps most.

`apply.py` then picks the cut on the `final` rows:
- **Rule `cut`:** drop a base row if p < τ. τ is the first maximiser of the official ranking F1 of the bundle's training-OOF final rows, scored against the training events with `--ignore-flagged`, over τ ∈ {0, 0.025, …, 0.7}. τ = 0 means keep everything.
- **Rule `cut2`:** drop if p < τ AND the detector score < σ, with σ ∈ {0.6, 0.7, 0.8, 0.9, 1.01}.

`cnn.py <set> full` trains on all training candidates and scores the base test rows. The tuned τ is applied to those scores. Test streams and val labels are read only for the final scoring, and for the clearly marked "analysis only" AUC in `tune.json`.

## Numbers

SR-CV is pooled over 4 subject-disjoint folds (253 evaluable falls). Promotion uses `srcv.promote` (5000 paired subject-cluster bootstrap resamples), against `final_norep` bag 0.

| variant | verifier seeds | rule | pooled rank F1 | Δ [95% CI] | P(Δ>0) | folds improved (F1 per fold) | TP/FP/FN @0.3 | val (TP/FP/FN) | promote |
|---|---|---|---|---|---|---|---|---|---|
| base `final_norep` b0 | – | – | 0.9139 | – | – | – (.912/.873/.909/.964) | 239/23/14 | 0.9373 (63/3/4) | – |
| `r18_cut` | 0 | cut | 0.9265 | +0.0126 [+0.000, +0.036] | 0.873 | **1/4** (.912/.920/.909/.964) | 239/16/14 | 0.9373 (63/3/4) | no |
| `r18_cut_s1` | 1 | cut | 0.9211 | +0.0072 [+0.000, +0.019] | 0.873 | 1/4 (fold 1 .899) | 239/19/14 | 0.9373 | no |
| `r18_cut_s2` | 2 | cut | 0.9229 | +0.0090 [+0.000, +0.024] | 0.873 | 1/4 (fold 1 .906) | 239/18/14 | 0.9373 | no |
| `r18x3_cut` | mean of 0,1,2 | cut | 0.9283 | +0.0144 [+0.000, +0.042] | 0.873 | 1/4 (fold 1 .927) | 239/15/14 | 0.9373 | no |
| `r18_cut2` | 0 | cut2 | 0.9302 | +0.0163 [+0.000, +0.048] | 0.873 | 1/4 (fold 1 .934) | 239/14/14 | 0.9373 | no |

Cut chosen on training OOF, and rows dropped on the test side (`results/<variant>/tune.json`):

| variant | fold 0 | fold 1 | fold 2 | fold 3 | official → val |
|---|---|---|---|---|---|
| `r18_cut`, τ / test rows dropped | 0 / 0 | 0.025 / 7 (0 hits) | 0.025 / 2 (2 flagged-fall hits, no score change) | 0.025 / 0 | 0.025 / 0 |
| `r18x3_cut`, τ / dropped | 0 / 0 | 0.125 / 8 (0 hits) | 0.025 / 1 | 0.1 / 1 | 0 / 0 |

### How well the verifier separates hits from false alarms

| set | AUC, all training candidates (inner OOF) | AUC on the hard rows (OOF final rows, 6–13 FPs per fold) | AUC on the test rows (analysis only) |
|---|---|---|---|
| fold 0 | 0.95 | 0.69 | 0.84 |
| fold 1 | 0.93 | 0.70 | 0.80 |
| fold 2 | 0.96 | 0.57 | 0.74 |
| fold 3 | 0.94 | 0.64 | 0.55 |
| official | 0.94 | 0.57 | 0.83 |

The detector score alone has an AUC of 0.64 on fold 0's OOF final rows, so the verifier is somewhat complementary. But at about 0.6–0.7 on the rows that matter, no cut can drop false positives without also dropping true positives in most folds. The training-OOF tuning therefore correctly picks τ ≈ 0 there.

### Screens on fold 0 only

These were not run on all folds because they were no better on fold 0's training OOF. Their probabilities are in `probs/`.

| screen | AUC on fold 0 OOF final rows | τ chosen on fold 0 |
|---|---|---|
| 3-channel frame-difference crop, seed 0 (`r18`) | 0.686 | 0 |
| same, 3-seed mean | 0.697 | 0 |
| 6 channels: + per-radar log amplitude (static profile), ± 5.5 s window (`r18c6`) | 0.605 | 0 |
| hard negatives only, no random windows (`r18hard`) | 0.706 | 0 |
| first run on the earlier crop files (`r18_oldcrops`, same design; fp16 rounding differs) | 0.713 | 0 |

### Where the base false alarms are

Post hoc, from the composition of the test streams. This was not used for any choice.
- **Fold 1:** 14 FPs. Most are inside ADL clips of one held-out ADL subject (`Laver_dents` ×9, `Lire_livre`, `Faire_the`, `Dormir`). The verifier gives most of them p < 0.06. This fold produces the whole gain.
- **Folds 0, 2 and 3:** the FPs are mostly mislocalised detections inside fall clips (tIoU 0–0.29), plus a few `Autre` and ADL windows. Dropping such a row can only trade an FP @0.3 for a lost TP @0.1, so a hit/no-hit verifier cannot help there.

## Reproduce

```bash
cd /home/user/challenge1/work/R3_verify
export PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src ABC_ROOT=/home/user/data_raw/abc2027_uwb_train_v1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
# weights: torchvision resnet18 ImageNet (BSD), sha256 f37072fd... -> /home/user/data_raw/r3/verify/resnet18-f37072fd.pth
curl -L -o /home/user/data_raw/r3/verify/resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
python3 maps.py fold0 fold1 fold2 fold3 official      # per-stream frame-diff maps (~610 MB, only needed by prep.py's crops)
python3 prep.py fold0 fold1 fold2 fold3 official      # cands/<set>_{train,test}.csv (+ legacy 3-ch crops)
python3 crops2.py fold0 fold1 fold2 fold3 official    # 6-ch crops used by cnn.py (~500 MB); maps/ and crops/ can then be deleted
./run_r18.sh fold0 fold1 fold2 fold3 official          # seed 0, inner + full  (~11 min per set, 1 thread)
./run_seeds.sh 1 fold0 fold1 fold2 fold3 official; ./run_seeds.sh 2 fold0 fold1 fold2 fold3 official
python3 apply.py r18 r18_cut --seeds 0; python3 apply.py r18 r18_cut2 --seeds 0 --rule cut2
python3 apply.py r18 r18_cut_s1 --seeds 1; python3 apply.py r18 r18_cut_s2 --seeds 2; python3 apply.py r18 r18x3_cut --seeds 0,1,2
python3 score.py r18_cut r18_cut2 r18_cut_s1 r18_cut_s2 r18x3_cut
# fold-0 screens: python3 cnn.py fold0 inner|full --tag r18c6 --half 55 --chans 012345 ; python3 cnn.py fold0 inner --tag r18hard --no-rand
```

The intermediate caches (`maps/`, `crops/`, `crops2/` under `/home/user/data_raw/r3/verify/`) were deleted at the end. Only the 45 MB weights file is kept.

`maps.py` comes from the first, interrupted attempt. The resumed run re-validated its 90 map files (shapes, logs) before using them.

## Files

- `maps.py`, `prep.py`, `crops2.py`: build the images and the candidate tables, per stream.
- `cnn.py`: the verifier.
- `apply.py`: picks the cut on training OOF and writes `results/<v>/{fold_k,val}.csv` and `tune.json`.
- `score.py`: runs the promotion rule and writes `results/<v>/promote.json`.
- `cands/`: candidate tables with labels.
- `probs/<tag>/`: verifier probabilities, per set, mode and seed.
- `logs/`: run logs.
- `results/base/`: a copy of the base CSVs.

## Compliance

- Every image is computed from one stream's own CIR, and the per-bin median is that stream's own. Nothing is compared or aligned across streams, with train/val segments, or within a stream to find repeats. No frame-repeat statistics are used.
- The pretrained weights are open-source (torchvision ImageNet, BSD-3). No other data source is used.
- Every threshold, epoch count and hyper-parameter was fixed before scoring or chosen on training-fold OOF. Test streams and val were used only for the final scoring and for the analysis-only AUC column.

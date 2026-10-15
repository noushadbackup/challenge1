#!/bin/bash
# restart after container restart (05:16): lane E = SR-CV fold 3 (full fit), then fallback train13 -> val
source /home/user/challenge1/work/FINAL/env.sh
cd $REPO; L=$REPO/work/FINAL/logs
echo "start $(date -Is)" > $L/laneE.status
python3 -m srcv.run_pipeline final_b0 --config uwbfall/configs/final.json --folds 3 > $L/srcv_final_b0_f3.log 2>&1
echo "srcv f3 rc=$? $(date -Is)" >> $L/laneE.status
python3 work/FINAL/rss.py python3 -m uwbfall train --config uwbfall/configs/fallback_physics.json --train-splits train --out /home/user/data_raw/uwbfall_models/fallback_physics_train13 > $L/train_fallback13.log 2>&1
echo "fb13 train rc=$? $(date -Is)" >> $L/laneE.status
python3 -m uwbfall predict --models /home/user/data_raw/uwbfall_models/fallback_physics_train13 --split val --out $REPO/work/FINAL/out/val_fallback_train13.csv > $L/predict_fallback13_val.log 2>&1
echo "fb13 val rc=$? $(date -Is)" >> $L/laneE.status

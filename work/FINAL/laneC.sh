#!/bin/bash
# after lane B: physics fallback on train13 -> val (official CLI)
source /home/user/challenge1/work/FINAL/env.sh
cd $REPO; L=$REPO/work/FINAL/logs
until grep -q "srcv val rc" $L/laneB.status; do sleep 15; done
echo "start $(date -Is)" > $L/laneC.status
python3 -m uwbfall train --config uwbfall/configs/fallback_physics.json --train-splits train --out /home/user/data_raw/uwbfall_models/fallback_physics_train13 > $L/train_fallback13.log 2>&1
echo "train rc=$? $(date -Is)" >> $L/laneC.status
python3 -m uwbfall predict --models /home/user/data_raw/uwbfall_models/fallback_physics_train13 --split val --out $REPO/work/FINAL/out/val_fallback_train13.csv > $L/predict_fallback13_val.log 2>&1
python3 -m uwbfall score --pred $REPO/work/FINAL/out/val_fallback_train13.csv --split val --out $REPO/work/FINAL/out/val_fallback_train13.json >> $L/predict_fallback13_val.log 2>&1
echo "val rc=$? $(date -Is)" >> $L/laneC.status

#!/bin/bash
# lane F = resume fold 1 predict, then official train13 -> val (clean), then fallback bundle train+val
source /home/user/challenge1/work/FINAL/env.sh
cd $REPO; L=$REPO/work/FINAL/logs
echo "start $(date -Is)" > $L/laneF.status
python3 work/FINAL/resume_fold.py final_b0 1 > $L/srcv_final_b0_f1_resume.log 2>&1
echo "f1 resume rc=$? $(date -Is)" >> $L/laneF.status
python3 -m srcv.run_pipeline final_b0 --config uwbfall/configs/final.json --val-only > $L/srcv_final_b0_val.log 2>&1
echo "srcv val rc=$? $(date -Is)" >> $L/laneF.status
python3 work/FINAL/rss.py python3 -m uwbfall train --config uwbfall/configs/fallback_physics.json --train-splits train,val --out /home/user/data_raw/uwbfall_models/fallback_physics > $L/train_fallback.log 2>&1
echo "fb train rc=$? $(date -Is)" >> $L/laneF.status

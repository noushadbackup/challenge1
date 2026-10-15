#!/bin/bash
# lane B: SR-CV clean run folds 0,1, then official train -> val
source /home/user/challenge1/work/FINAL/env.sh
cd $REPO
L=$REPO/work/FINAL/logs
echo "start $(date -Is)" > $L/laneB.status
python3 -m srcv.run_pipeline final_b0 --config uwbfall/configs/final.json --folds 0,1 > $L/srcv_final_b0_f01.log 2>&1
echo "srcv f01 rc=$? $(date -Is)" >> $L/laneB.status
python3 -m srcv.run_pipeline final_b0 --config uwbfall/configs/final.json --val-only > $L/srcv_final_b0_val.log 2>&1
echo "srcv val rc=$? $(date -Is)" >> $L/laneB.status

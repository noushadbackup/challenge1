#!/bin/bash
# lane A: final train+val fit (18 streams), then SR-CV clean run folds 2,3
source /home/user/challenge1/work/FINAL/env.sh
cd $REPO
L=$REPO/work/FINAL/logs
echo "start $(date -Is)" > $L/laneA.status
python3 -m uwbfall train --config uwbfall/configs/final.json --train-splits train,val \
   --out /home/user/data_raw/uwbfall_models/final > $L/train_final.log 2>&1
echo "train_final rc=$? $(date -Is)" >> $L/laneA.status
python3 -m srcv.run_pipeline final_b0 --config uwbfall/configs/final.json --folds 2,3 > $L/srcv_final_b0_f23.log 2>&1
echo "srcv f23 rc=$? $(date -Is)" >> $L/laneA.status

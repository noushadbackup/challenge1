#!/bin/bash
# after lane A: physics fallback bundle on train+val (release-day switch)
source /home/user/challenge1/work/FINAL/env.sh
cd $REPO; L=$REPO/work/FINAL/logs
until grep -q "srcv f23 rc" $L/laneA.status; do sleep 15; done
echo "start $(date -Is)" > $L/laneD.status
python3 -m uwbfall train --config uwbfall/configs/fallback_physics.json --train-splits train,val --out /home/user/data_raw/uwbfall_models/fallback_physics > $L/train_fallback.log 2>&1
echo "train rc=$? $(date -Is)" >> $L/laneD.status

#!/bin/bash
set -x
S=/tmp/claude-0/-home-user-challenge1/f892ab69-5d71-5ee6-a35c-b07836a0af8f/scratchpad
C=/home/user/data_raw/audit_clone; O=/home/user/data_raw/audit_out
export UWBFALL_CACHE=/home/user/data_raw/audit_cache
MK="make -C $C REPO=$C MODELS=/home/user/data_raw/audit_models SUB=$O/sub"
mkdir -p $O/sub
python3 $S/meas.py $O/meas_features.json $MK features > $O/log_features.txt 2>&1
python3 $S/meas.py $O/meas_reproduce_val.json $MK reproduce-val > $O/log_reproduce_val.txt 2>&1
python3 $S/meas.py $O/meas_train_final.json $MK train-final > $O/log_train_final.txt 2>&1
echo LANE2 DONE

#!/bin/bash
# variant r18 (3-channel frame-difference crops, +-4 s, 6 epochs, seed 0) on folds 1-3 + official; fold0 already run
cd /home/user/challenge1/work/R3_verify
export PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src ABC_ROOT=/home/user/data_raw/abc2027_uwb_train_v1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
for S in "$@"; do
  for m in inner full; do
    [ -f probs/r18/${S}_${m}_s0.csv ] || nice -n 10 python3 cnn.py $S $m --tag r18 > logs/r18_${S}_${m}.log 2>&1
  done
done

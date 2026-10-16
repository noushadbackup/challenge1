#!/bin/bash
# extra TCN-independent verifier seeds for variant r18: ./run_seeds.sh <seed> <set> [<set> ...]
cd /home/user/challenge1/work/R3_verify
export PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src ABC_ROOT=/home/user/data_raw/abc2027_uwb_train_v1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
seed=$1; shift
for S in "$@"; do
  for m in inner full; do
    [ -f probs/r18/${S}_${m}_s${seed}.csv ] || nice -n 10 python3 cnn.py $S $m --tag r18 --seed $seed > logs/r18_${S}_${m}_s${seed}.log 2>&1
  done
done

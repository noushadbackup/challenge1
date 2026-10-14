#!/bin/bash
# two lanes of prep (2 threads each)
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
(for t in 0 2 official; do python3 work/I1_clip/prep.py $t --hgb3; done) > work/I1_clip/logs/prep_a.log 2>&1 &
(for t in 1 3; do python3 work/I1_clip/prep.py $t --hgb3; done) > work/I1_clip/logs/prep_b.log 2>&1 &
wait

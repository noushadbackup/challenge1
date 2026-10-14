#!/bin/bash
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
for b in uwb_default tcn_bag1 tcn_bag2; do python3 work/I1_clip/variants.py $b 0,1,2,3 D; done

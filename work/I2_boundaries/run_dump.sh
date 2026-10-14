#!/bin/bash
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
(python3 work/I2_boundaries/dump.py 1,2,3,official --parts test; python3 work/I2_boundaries/dump.py 0,1 --parts train) > work/I2_boundaries/logs/dump_a.log 2>&1 &
(python3 work/I2_boundaries/dump.py 2,3,official --parts train) > work/I2_boundaries/logs/dump_b.log 2>&1 &
wait

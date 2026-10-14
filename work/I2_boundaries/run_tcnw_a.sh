#!/bin/bash
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
python3 work/I2_boundaries/tcnw.py tcnW_bag0 0 ${1:-0,1,official} > work/I2_boundaries/logs/tcnw_${2:-a}.log 2>&1

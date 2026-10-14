#!/bin/bash
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
python3 work/I2_boundaries/hgbvar.py official,0,1,2 Z,W 2>&1 | grep -v Warn > work/I2_boundaries/logs/hgbvar.log

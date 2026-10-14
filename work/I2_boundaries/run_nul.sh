#!/bin/bash
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
(python3 work/I2_boundaries/hgbvar.py 0,1,2,3,official Nul && python3 work/I2_boundaries/decvar.py 0,1,2,3,official Nul --hgb Nul) 2>&1 | grep --line-buffered -v Warn > work/I2_boundaries/logs/nul.log

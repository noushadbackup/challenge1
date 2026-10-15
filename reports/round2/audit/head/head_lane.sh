#!/bin/bash
set -x
S=/tmp/claude-0/-home-user-challenge1/f892ab69-5d71-5ee6-a35c-b07836a0af8f/scratchpad
C=/home/user/data_raw/audit_clone_head; PKG=/home/user/data_raw/abc2027_uwb_train_v1; O=/home/user/data_raw/audit_out/head
T=/home/user/data_raw/fake_test/streams/test
export PYTHONDONTWRITEBYTECODE=1 ABC_ROOT=$PKG PYTHONPATH=$C:$PKG/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /tmp
rm -rf /home/user/data_raw/audit_cache_head
sync; echo 3 > /proc/sys/vm/drop_caches
UWBFALL_CACHE=/home/user/data_raw/audit_cache_head python3 $S/meas.py $O/meas_cold.json python3 -m uwbfall predict --models $C/models/final --sources dir:$T:test --out $O/pred_cold.csv > $O/log_cold.txt 2>&1
UWBFALL_CACHE=/home/user/data_raw/audit_cache_head python3 $S/meas.py $O/meas_warm.json python3 -m uwbfall predict --models $C/models/final --sources dir:$T:test --out $O/pred_warm.csv > $O/log_warm.txt 2>&1
UWBFALL_CACHE=/home/user/data_raw/audit_cache_head python3 $S/meas.py $O/meas_warm_dataraw.json python3 -m uwbfall predict --models /home/user/data_raw/uwbfall_models/final --sources dir:$T:test --out $O/pred_warm_dataraw.csv > $O/log_warm_dataraw.txt 2>&1
echo LANE DONE

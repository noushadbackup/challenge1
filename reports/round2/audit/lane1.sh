#!/bin/bash
set -x
S=/tmp/claude-0/-home-user-challenge1/f892ab69-5d71-5ee6-a35c-b07836a0af8f/scratchpad
C=/home/user/data_raw/audit_clone; PKG=/home/user/data_raw/abc2027_uwb_train_v1; O=/home/user/data_raw/audit_out
T=/home/user/data_raw/fake_test/streams/test
export PYTHONDONTWRITEBYTECODE=1 ABC_ROOT=$PKG PYTHONPATH=$C:$PKG/src
run() { name=$1; cache=$2; bundle=$3; shift 3
  sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
  UWBFALL_CACHE=$cache python3 $S/meas.py $O/meas_$name.json env "$@" python3 -m uwbfall predict --models $bundle --sources dir:$T:test --out $O/pred_$name.csv > $O/log_$name.txt 2>&1
}
cd /home/user/data_raw
rm -rf audit_cache_p2 audit_cache_p1 audit_cache_p4
run t2_cold audit_cache_p2 /home/user/data_raw/uwbfall_models/final OMP_NUM_THREADS=2
run t2_warm audit_cache_p2 /home/user/data_raw/uwbfall_models/final OMP_NUM_THREADS=2
run t2_warm_omp4env audit_cache_p2 /home/user/data_raw/uwbfall_models/final OMP_NUM_THREADS=4
run t2_repocopy audit_cache_p2 $C/models/final OMP_NUM_THREADS=2
run t1_cold audit_cache_p1 /home/user/data_raw/audit_bundles/final_t1 OMP_NUM_THREADS=1
run t4_cold audit_cache_p4 /home/user/data_raw/audit_bundles/final_t4 OMP_NUM_THREADS=4
echo LANE1 DONE

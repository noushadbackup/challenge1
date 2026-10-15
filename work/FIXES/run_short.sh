#!/bin/bash
# Predict each short fixture with the FINAL bundle, then run the release-day checker on it.
# usage: run_short.sh <fixtures_root> <bundle> [tag]
ROOT=$1; BUNDLE=$2; TAG=${3:-final}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 UWBFALL_CACHE=/home/user/data_raw/fixes_cache \
       PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
for d in $(ls -d $ROOT/L* | sort -V); do
  echo "=== $(basename $d)"
  python3 -m uwbfall predict --models $BUNDLE --sources dir:$d/test:test --out $d/pred_$TAG.csv > $d/predict_$TAG.log 2>&1
  rc=$?
  echo "predict rc=$rc"; [ $rc -ne 0 ] && tail -6 $d/predict_$TAG.log
  [ -f $d/pred_$TAG.csv ] && { cat $d/pred_$TAG.csv; python3 -m uwbfall.tests.check_submission $d/pred_$TAG.csv $d/test test_04 | tail -2; }
done

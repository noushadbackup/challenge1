#!/bin/bash
# After the extra bags are fitted: derive candidates, jitter replays, variants, table and promotion checks (3 bags).
# Targets 0,1,2 of the extra bags are derived early (as soon as those folds are fitted); 3 and official afterwards.
set -e
cd /home/user/challenge1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
L=work/R2_EVAL/logs
for b in tcn_bag1 tcn_bag2; do
  python3 work/R2_EVAL/derive.py $b 0,1,2 > $L/derive_${b}_012.log 2>&1
  python3 work/R2_EVAL/jitter.py $b C3,C4 0,1,2 > $L/jitter_${b}_012.log 2>&1
done
for b in tcn_bag1 tcn_bag2; do
  until [ -f /home/user/data_raw/srcv/runs/$b/official_train/meta.json ]; do sleep 20; done
done
( python3 work/R2_EVAL/derive.py tcn_bag1 3,official > $L/derive_bag1.log 2>&1 && python3 work/R2_EVAL/jitter.py tcn_bag1 C3,C4 3,official > $L/jitter_bag1.log 2>&1 ) &
( python3 work/R2_EVAL/derive.py tcn_bag2 3,official > $L/derive_bag2.log 2>&1 && python3 work/R2_EVAL/jitter.py tcn_bag2 C3,C4 3,official > $L/jitter_bag2.log 2>&1 ) &
wait
python3 work/R2_EVAL/variants.py uwb_default,tcn_bag1,tcn_bag2
python3 work/R2_EVAL/score_all.py table C0,C1,C2,C3,C4,C3K,C3J,C3JK,C4K,C4J,C4JK,C3G3,C4G3 uwb_default,tcn_bag1,tcn_bag2 > $L/table3.log 2>&1
for p in "C2 C3" "C1 C2" "C1 C4" "C0 C3" "C4 C3" "C3 C3J" "C4 C4J" "C3 C3K" "C3 C3JK" "C4 C4K" "C4 C4JK" "C0 C3J"; do
  python3 work/R2_EVAL/score_all.py promote $p uwb_default,tcn_bag1,tcn_bag2 2>&1 | tail -1 >> $L/promote3.log
done
echo ALLDONE >> $L/promote3.log

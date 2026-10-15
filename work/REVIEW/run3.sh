export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/user/challenge1:/home/user/data_raw/abc2027_uwb_train_v1/src
S=/tmp/claude-0/-home-user-challenge1/f892ab69-5d71-5ee6-a35c-b07836a0af8f/scratchpad/rev
UWBFALL_CACHE=/home/user/data_raw/review_fresh/cache python3 $S/fresh3.py fresh
python3 $S/fresh3.py orig
python3 $S/fresh3.py val
echo DONE

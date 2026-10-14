#!/bin/sh
# Full E2 pipeline (~12 min on 2 cores). Outputs in ./out
set -e
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
python3 s1_proto.py                                                    # prototype 'full': train OOF + val candidates (~1 min)
python3 s2_replay.py train out/train_oof_proto.csv out/replay_train.csv &  # annotator replay on 37 crop variants (~6 min)
python3 s2_replay.py val out/val_proto.csv out/replay_val.csv
wait
python3 s3_select.py                     # select on train OOF only, then score frozen configs once on val (official CLI)
python3 s4_diag.py val out/val_B_replay_post.csv B_replay_post          # per-type hit rates / signed errors
python3 s5_paired.py val out/val_proto_cli.csv out/val_B_replay_post.csv # paired bootstrap

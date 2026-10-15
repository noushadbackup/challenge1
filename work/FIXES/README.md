# Round-2 fixer: scripts and logs

See `reports/round2/FIXES.md` for the fixes and their before/after evidence.

- `make_short.py <out_root> <src_h5> <start_s> <len_s>...` crops a published stream into single-stream folders named
  `test`, one per length.
- `run_short.sh <root> <bundle>` predicts each fixture (cache in `/home/user/data_raw/fixes_cache`), then runs
  `check_submission`.
- `logs/` holds the logs of every verification quoted in FIXES.md.
- `val_after_official.json` is the official CLI report for the val CSV after the fixes.

Scratch outputs are outside git, in `/home/user/data_raw/fixes_{sub,short,short2,mixed,cache}` (about 50 MB). They can be
deleted.

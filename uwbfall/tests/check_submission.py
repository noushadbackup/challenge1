"""Release-day submission checks (no labels needed).

    python3 -m uwbfall.tests.check_submission <submission.csv> <folder with stream_XX.h5> [expected ids, e.g. test_00,...]

1. official abc2027.evaluate.validate_predictions against the official stream durations of the folder;
2. exact header stream_id,t_start,t_end,score,t_alarm;
3. every stream of the folder has >= 1 row and no unknown ids (expected ids default: <folder name>_XX);
4. sanity: per-stream events/h within [10, 150] (train/val truth: ~58 falls/h), scores in [0, 1], durations 0.2-30 s.
Exit code 0 = all hard checks pass (1-3); 4 prints WARN only.
"""
import csv
import sys
from pathlib import Path

from abc2027 import evaluate as E

pred, folder = Path(sys.argv[1]), Path(sys.argv[2])
dur = E.stream_durations(folder)
exp = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(dur)
recs = E.validate_predictions(pred, dur)                      # raises ValidationError on any bad row
with pred.open() as f:
    header = next(csv.reader(f))
hard = []
if header != ["stream_id", "t_start", "t_end", "score", "t_alarm"]:
    hard.append(f"header {header}")
ids = sorted({r["stream_id"] for r in recs})
if sorted(dur) != sorted(exp):
    hard.append(f"folder ids {sorted(dur)} != expected {exp}")
missing = [s for s in exp if s not in ids]
if missing:
    hard.append(f"streams without any prediction: {missing}")
print(f"{pred}: {len(recs)} rows, {len(ids)} streams, official validate_predictions OK")
for s in sorted(dur):
    rows = [r for r in recs if r["stream_id"] == s]
    rate = len(rows) / (dur[s] / 3600)
    d = [r["t_end"] - r["t_start"] for r in rows]
    sc = [r["score"] for r in rows]
    warn = []
    if not 10 <= rate <= 150:
        warn.append("events/h outside [10, 150]")
    if rows and (min(sc) < 0 or max(sc) > 1):
        warn.append("score outside [0, 1]")
    if rows and (min(d) < 0.2 or max(d) > 30):
        warn.append("event duration outside [0.2, 30] s")
    print(f"  {s}: {len(rows):3d} rows, {rate:5.1f} events/h, {dur[s] / 60:5.1f} min"
          + (f", dur {min(d):.2f}-{max(d):.2f} s, score {min(sc):.3f}-{max(sc):.3f}" if rows else "")
          + ("  WARN: " + "; ".join(warn) if warn else ""))
print("SUBMISSION CHECK:", "OK" if not hard else "FAILED " + " | ".join(hard))
sys.exit(1 if hard else 0)

import sys, csv, hashlib
from pathlib import Path
from abc2027.evaluate import validate_predictions, stream_durations
import pandas as pd
p, folder = Path(sys.argv[1]), Path(sys.argv[2])
dur = stream_durations(folder)
recs = validate_predictions(p, dur)
raw = p.read_bytes()
hdr = raw.split(b"\n",1)[0].decode()
df = pd.read_csv(p)
print(f"file={p} sha256={hashlib.sha256(raw).hexdigest()[:16]} rows={len(recs)} header={hdr!r} CRLF={b'\\r' in raw}")
print("durations", {k: round(v,1) for k,v in dur.items()})
print("per-stream rows", df.stream_id.value_counts().sort_index().to_dict())
print("missing streams", sorted(set(dur)-set(df.stream_id)))
print("score range", df.score.min(), df.score.max(), "dup rows", int(df.duplicated().sum()))
al = df.dropna(subset=["t_alarm"])
print("t_alarm present", len(al), "t_alarm in [t_start,t_end+0.001]", int(((al.t_alarm>=al.t_start-1e-9)&(al.t_alarm<=al.t_end+1e-3)).sum()))
print("dur s: min/med/max", df.eval("t_end-t_start").min(), df.eval("t_end-t_start").median(), df.eval("t_end-t_start").max())
print("validate_predictions: OK")

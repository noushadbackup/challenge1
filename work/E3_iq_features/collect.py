"""Collect results/*.json into a markdown table (results/summary.md + results/summary.json)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for p in sorted((HERE / "results").glob("*.json")):
    if p.name == "summary.json":
        continue
    r = json.load(open(p))
    rows.append(r)
hdr = "| tag | n_feat | train OOF rank F1 (mean of reps) | per rep | val rank F1 | val F1@.1/.3/.5/.7 | val TP/FP/FN@.3 | val med start err | val FA/h |"
lines = [hdr, "|" + "---|" * 9]
for r in rows:
    tag = r["arm"] + ("_es_s%d" % r["seed"] if r["early_stopping"] else "")
    lines.append(f"| {tag} | {r['n_features']} | {r['train_oof_rank_f1']} | {r['train_oof_rank_f1_per_rep']} | {r.get('val_rank_f1', '-')} | "
                 f"{r.get('val_f1@0.1/0.3/0.5/0.7', '-')} | {r.get('val_tp_fp_fn@0.3', '-')} | {r.get('val_start_err_median@0.3', '-')} | {r.get('val_fa_per_hour@0.3', '-')} |")
(HERE / "results" / "summary.md").write_text("\n".join(lines) + "\n")
json.dump([{k: v for k, v in r.items() if k not in ("train_oof_per_stream", "features")} for r in rows],
          open(HERE / "results" / "summary.json", "w"), indent=1)
print("\n".join(lines))

"""Markdown tables for the I3 report from out/table.json and out/promote_*.json.
    python3 work/I3_score/mdtable.py V1,V2,...
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
t = json.loads((OUT / "table.json").read_text())
print("| variant | pooled rank F1 ± sd | 95% CI | folds 0/1/2/3 | F1 @.1/.3/.5 | TP/FP/FN @0.3 | FA/h | med start err | val 3-bag mean (b1/b2/b0) |")
print("|---|---|---|---|---|---|---|---|---|")
for v in sys.argv[1].split(","):
    r = t[v]; m = r["seedmean"]; p = m["pooled"]
    ci = r["bootstrap_seedavg_counts"]
    ci = f"[{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]" if isinstance(ci, dict) and "ci95" in ci else str(ci)[:30]
    vals = "/".join(f"{x['val']['rank_f1']:.3f}" for x in r["per_bag"])
    print(f"| {v} | {p['rank_f1']:.4f} ± {m['pooled_rank_sd_over_bags']:.3f} | {ci} | {'/'.join(f'{x:.3f}' for x in m['folds'])} | "
          f"{p['f1@0.1']:.3f}/{p['f1@0.3']:.3f}/{p['f1@0.5']:.3f} | {p['tp@0.3']:.1f}/{p['fp@0.3']:.1f}/{p['fn@0.3']:.1f} | "
          f"{p['fa_h@0.3']:.2f} | {p['start_med@0.3']:.3f} s | {m['val']['rank_f1']:.4f} ({vals}) |")
print()
print("| A → B | Δ [95% CI] | P(Δ>0) | folds improved (per-fold Δ) | val A → B | decision |")
print("|---|---|---|---|---|---|")
for f in sorted(OUT.glob("promote_*.json")):
    r = json.loads(f.read_text())
    keys = list(r)
    d = r["delta"]; ci = r["delta_ci95"]
    pf = [x["delta"] if isinstance(x, dict) else x for x in r["per_fold"]]
    print(f"| {r['A']} → {r['B']} | {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {r['p_delta_gt_0']:.3f} | "
          f"{r['n_folds_improved']}/4 ({', '.join(f'{x:+.4f}' for x in pf)}) | "
          f"{r.get('val_A', 0):.4f} → {r.get('val_B', 0):.4f} | {'PASS' if r.get('promote') else 'FAIL'} |")

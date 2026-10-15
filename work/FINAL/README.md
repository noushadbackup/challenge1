# work/FINAL: round-2 integration (frozen config, clean SR-CV run, final models, dry run)

Report: `reports/round2/FINAL.md`. Release-day runbook: `SUBMISSION.md`. Everything is driven by the root `Makefile`.

| file | purpose |
|---|---|
| `env.sh` | environment: PYTHONPATH, 2 threads |
| `laneA.sh`, `laneB.sh` | first attempt: final train+val fit, SR-CV clean run (killed by a container restart at about 05:16) |
| `laneE.sh`, `laneF.sh` | restart: fold 3, fold-1 predict resume, official train→val, both physics fallback bundles |
| `laneC.sh`, `laneD.sh` | first-attempt fallback lanes. They never ran and were superseded by lanes E/F |
| `resume_fold.py` | predict an SR-CV fold from its completed bundle (`srcv.run_pipeline` semantics) |
| `score.py link\|check\|table\|promote` | FINAL vs C3 table + promotion rule (3 bags), using the R2_EVAL `score_all` tooling (official evaluator, val via CLI) |
| `ship.py <bundle> <name>` | copy a bundle's runnable files to `models/<name>/` with `MANIFEST.sha256` and `manifest.json` |
| `rss.py <cmd>` | wall time + peak RSS of a command |
| `logs/` | all run logs (`*_r2.log` = this session's dry runs and checks) |
| `out/` | `table.json`, `promote_C3_vs_FINAL_{3bags,bag0}.json`, fallback val CSV and score |
| `preds/<bag>/{C3,FINAL}/` | symlinks: FINAL bag `uwb_default` = `srcv/results/final_b0` (clean run), other bags = I1 G7N4 |

Bundles live outside git: `/home/user/data_raw/uwbfall_models/{final,fallback_physics,fallback_physics_train13,final_train13 -> srcv/runs/final_b0/official_train}`
and `/home/user/data_raw/srcv/runs/final_b0/fold_k`.

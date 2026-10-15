# challenge1: ABC 2027 UWB fall-event detection

Raw 3-radar UWB streams go in; one CSV row per detected fall comes out (`stream_id,t_start,t_end,score,t_alarm`).

| you want to | read / run |
|---|---|
| produce the test submission on release day | [`SUBMISSION.md`](SUBMISSION.md): `make predict-test TEST_DIR=<folder named test>` |
| install the pipeline / use its CLI and API | [`uwbfall/README.md`](uwbfall/README.md): `pip install ./uwbfall` (exact versions: [`requirements-lock.txt`](requirements-lock.txt)) |
| see the evidence for the final system | [`reports/round2/FINAL.md`](reports/round2/FINAL.md) (SR-CV table, promotion rule, dry run) |
| check reproducibility and rule compliance | [`reports/round2/AUDIT.md`](reports/round2/AUDIT.md) |
| understand the method and plan | [`reports/REPORT.md`](reports/REPORT.md), [`reports/PLAN.md`](reports/PLAN.md) |

Key facts:
- Frozen config: `uwbfall/configs/final.json`. Shipped models with sha256 manifests: `models/final/` and the physics-only fallback in `models/fallback_physics/`.
- The challenge package (`/home/user/data_raw/abc2027_uwb_train_v1`) is read-only input and is never written to.
- Run `make` targets from this folder or with `make -f <clone>/Makefile`. The Makefile locates the repo from its own path.
- Predictions are bit-reproducible with the locked versions and the bundle's `runtime.threads = 2`.

# Running the full pipeline on your own PC

Goal: run feature extraction, training, validation and the submission dry run on your own machine, so Claude
credits are only spent on reading results. Everything below runs on **CPU**. The TCN does not use the GPU yet
(`uwbfall` has no `runtime.device` option); adding it is a separate, optional step.

## 1. System

* **Windows:** use **WSL2 + Ubuntu 22.04/24.04**. The code, paths and Makefile assume Linux.
* **Disk:** about 30 GB free (8.2 GB zip + 12 GB unpacked + about 3 GB caches/models + the test release later).
* **RAM:** 16 GB is plenty (peak about 1.6 GB per step).
* **Tools:** `git`, `make`, `unzip`, `curl` (`sudo apt install git make unzip curl`).

## 2. Same folder layout as the cloud machine

The Makefile and `srcv/` default to `/home/user/data_raw`, so recreate that layout:

```bash
sudo mkdir -p /home/user && sudo chown "$USER" /home/user
mkdir -p /home/user/data_raw
```

## 3. Code

```bash
cd /home/user
# private repo: log in first (gh auth login, or a personal access token)
git clone -b claude/nice-cori-k0o5eu https://github.com/noushadbackup/challenge1.git
cd challenge1
```

## 4. Python environment (exact versions)

The shipped models were trained with Python 3.13 and the exact versions in `requirements-lock.txt`. Pickled
scikit-learn/LightGBM models need those versions. The easiest way to get Python 3.13 on Ubuntu is `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
cd /home/user/challenge1
uv venv -p 3.13 .venv && source .venv/bin/activate
uv pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install gdown
```

Activate the environment (`source /home/user/challenge1/.venv/bin/activate`) in every new terminal before running `make`.

## 5. Dataset (8.2 GB)

```bash
cd /home/user/data_raw
gdown 1gJfL1YtObCs9JBFjPrC8qYq9a7L2kt2u -O abc2027_uwb_train_v1.zip   # or download from the challenge site
unzip -q abc2027_uwb_train_v1.zip && rm abc2027_uwb_train_v1.zip
ls abc2027_uwb_train_v1/streams/train | head -3   # stream_00.h5 ...
```

The data is CC BY-NC 4.0. Only this package may be used; the original LIARA recordings are not allowed.

## 6. Full train / validation run (in this order)

Run these from `/home/user/challenge1`. Times are from the cloud machine (4 CPU cores, 2 threads); a desktop CPU is usually faster.

| step | command | time | expected result |
|---|---|---|---|
| 1. features | `make features` | ~10 min | feature cache for 18 streams in `/home/user/data_raw/uwbfall_cache` |
| 2. train on train → score val | `make reproduce-val` | ~20 min | official val ranking F1 **0.952381** (64 TP / 2 FP / 3 FN at θ=0.3) |
| 3. final model on train + val | `make train-final` | ~22 min | bundle in `/home/user/data_raw/uwbfall_models/final` |
| 4. compare with the shipped model | `make verify-models` | seconds | OK (only `meta.json` timing fields may differ) |
| 5. sanity checks | `make checks` | ~1 min | configs OK, drift OK, SHORT-STREAM OK |
| 6. submission rehearsal | `make dry-run` | ~5 min | `SUBMISSION CHECK: OK`, 0 drift alerts |
| 7. (optional) subject-disjoint CV | `make srcv` | ~1.5–2 h | pooled SR-CV ≈ 0.918 (bag 0); the 0.922 headline is a 3-bag mean |

Exact bit-for-bit equality needs the same versions and 2 threads (the Makefile sets that). On a different CPU the
last digits of the TCN can differ. A score within about ±0.01 of the expected value is normal; a big difference is not.

## 7. Release day (test set)

```bash
# put the 5 test files as stream_00.h5 ... stream_04.h5 in a folder named "test"
make predict-test TEST_DIR=/path/to/streams/test
# -> /home/user/data_raw/submission/submission.csv  (+ official validation, drift monitors, checks)
# only if the drift monitors raise an alert:
make predict-test-fallback TEST_DIR=/path/to/streams/test
```

Full runbook: `SUBMISSION.md`.

## 8. What to send back to Claude (keep it small)

Do not paste whole logs. Send only the final score line(s) that `make reproduce-val` / `make srcv` print, any
error message, and the wall time. That is enough for the next decision and costs very few credits.

## GPU (optional, later)

The RTX 5060 Ti would mainly speed up the TCN part (about 15 of the ~20 min training). Using it needs a code change
(`runtime.device = "cuda"`) and PyTorch with CUDA 12.8 (`--index-url https://download.pytorch.org/whl/cu128`).
GPU results are not bit-identical to the CPU ones, so keep the CPU setup above for the official submission.

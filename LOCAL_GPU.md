# Running the pipeline on your own PC (RTX 5060 Ti 16 GB)

Goal: run the heavy training and experiments on your machine for free, so Claude credits are only
spent on reading the results and deciding the next step.

> Status: the setup below works today on CPU. GPU training for the TCN (`runtime.device = "cuda"`)
> will be added to `uwbfall` right after round 2 finishes. Round-2 agents are editing `uwbfall/models.py`
> right now, so changing it in parallel would cause conflicts. Steps 1–6 are the same either way.

## 1. System

* **Windows:** use **WSL2 + Ubuntu 22.04/24.04**. The code and paths assume Linux.
  NVIDIA's Windows driver also provides CUDA inside WSL2; do **not** install a Linux driver inside WSL.
* **NVIDIA driver:** R570 or newer. The 5060 Ti is a Blackwell GPU (sm_120), and older drivers do not support it.
* **Disk:** about 30 GB free (8.2 GB zip + 12 GB unpacked + caches).
* **RAM:** 16 GB is enough. Every step stays below about 5 GB.

## 2. Same folder layout as the cloud machine

Some scripts (`srcv/`) still use absolute paths under `/home/user/data_raw`, so recreate that layout:

```bash
sudo mkdir -p /home/user && sudo chown "$USER" /home/user
mkdir -p /home/user/data_raw
```

## 3. Code

```bash
cd /home/user
git clone -b claude/nice-cori-k0o5eu https://github.com/noushadbackup/challenge1.git   # private repo: log in with gh auth login or a token first
cd challenge1
```

## 4. Python environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# PyTorch with CUDA 12.8: the first builds that support Blackwell / RTX 50xx (sm_120)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pandas scikit-learn lightgbm xgboost catboost ruptures h5py matplotlib threadpoolctl gdown
```

Check the GPU (all three lines must look right):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
# expect: True, 'NVIDIA GeForce RTX 5060 Ti', and 'sm_120' in the list
```

## 5. Dataset (8.2 GB)

Download it from the challenge site (*Dataset and Tutorial* page), or from the command line:

```bash
cd /home/user/data_raw
gdown 1gJfL1YtObCs9JBFjPrC8qYq9a7L2kt2u -O abc2027_uwb_train_v1.zip
unzip -q abc2027_uwb_train_v1.zip && rm abc2027_uwb_train_v1.zip
ls abc2027_uwb_train_v1/streams/train | head -3   # stream_00.h5 ...
```

The data is CC BY-NC 4.0. Only this package may be used; the original LIARA recordings are not allowed.

## 6. Check that everything reproduces (CPU, about 20 min)

```bash
cd /home/user/challenge1
export PYTHONPATH=$PWD
python3 -m uwbfall train   --config uwbfall/configs/default.json --train-splits train --out /home/user/data_raw/uwbfall_models/default_train13
python3 -m uwbfall predict --models /home/user/data_raw/uwbfall_models/default_train13 --split val --out /home/user/data_raw/val_pred.csv
python3 -m uwbfall score   --pred /home/user/data_raw/val_pred.csv --split val
```

Expected: val ranking F1 ≈ **0.9646** (65 TP / 0 FP / 2 FN at θ=0.3). On another machine the TCN can differ in the
last digits (different torch build or thread count). That is normal; a large difference is not.

## 7. What to send back (keep it small)

Do not paste logs. Send only the JSON summary that `score` prints (or the file given with `--out`), plus the
wall time. That is enough for the next decision and costs very few credits.

## Speed reference

| step | cloud (4 CPU) | your PC (expected) |
|---|---|---|
| feature extraction, 18 streams | ~15 min | ~5–10 min (CPU and disk bound) |
| TCN training (3 seeds + CV) | ~15 min | ~1–3 min on GPU (after GPU support lands) |
| one full SR-CV evaluation (4 folds) | ~1–1.5 h | ~15–25 min on GPU |

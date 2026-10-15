#!/usr/bin/env bash
# One command for the whole local run: setup -> dataset -> features -> train/val -> final model -> checks -> dry run.
#
#   bash run_all.sh            # CPU (bit-exact with the audited models)
#   bash run_all.sh cuda       # TCN on the GPU (RTX 50xx: installs a CUDA 12.8 torch build in .venv-gpu)
#   bash run_all.sh cuda --srcv          # also run the subject-disjoint CV (long: ~1.5-2 h on CPU)
#   bash run_all.sh cpu --skip-setup     # environment and dataset already in place
#   bash run_all.sh cuda --data /mnt/e/uwbfall/data_raw   # keep dataset, caches and models on another drive
#                                                       # (Windows E:\ = /mnt/e in WSL; "E:\uwbfall\data_raw" also works)
#
# Safe to re-run: finished steps are cheap (feature cache, existing dataset/venv are reused).
# Logs: /home/user/data_raw/run_all_logs/<step>.log   Summary at the end.
set -euo pipefail

DEVICE="${1:-cpu}"; shift || true
RUN_SRCV=0; SKIP_SETUP=0; DATA_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --srcv) RUN_SRCV=1 ;;
    --skip-setup) SKIP_SETUP=1 ;;
    --data) DATA_DIR="${2:?--data needs a folder}"; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
  shift
done
# Windows path (E:\uwbfall\data_raw or E:/uwbfall/data_raw) -> WSL path (/mnt/e/uwbfall/data_raw)
if [[ "$DATA_DIR" =~ ^([A-Za-z]):[\\/]?(.*)$ ]]; then
  drive="${BASH_REMATCH[1],,}"; rest="${BASH_REMATCH[2]//\\//}"
  DATA_DIR="/mnt/$drive/$rest"
fi
case "$DEVICE" in cpu|cuda|auto) ;; *) echo "device must be cpu, cuda or auto (got '$DEVICE')"; exit 1 ;; esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=/home/user/data_raw
PKG=$DATA/abc2027_uwb_train_v1
MODELS=$DATA/uwbfall_models
LOGS=$DATA/run_all_logs
DRIVE_ID=1gJfL1YtObCs9JBFjPrC8qYq9a7L2kt2u
# Python environments live in the Linux home (fast; Windows drives do not handle venv symlinks well).
if [ "$DEVICE" = cpu ]; then VENV=$HOME/.venvs/uwbfall-cpu; else VENV=$HOME/.venvs/uwbfall-gpu; fi

say()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*"; exit 1; }
declare -a SUMMARY=()
step() {  # step <name> <command...>   runs with a log file, records wall time, stops on failure
  local name=$1; shift
  local t0=$SECONDS
  say "$name"
  if ! "$@" 2>&1 | tee "$LOGS/$name.log"; then
    echo "---- last 30 lines of $LOGS/$name.log ----"; tail -30 "$LOGS/$name.log"
    die "step '$name' failed (full log: $LOGS/$name.log)"
  fi
  SUMMARY+=("$(printf '%-16s %5d s' "$name" $((SECONDS - t0)))")
}

# ---------------------------------------------------------------- 0. folders and tools
[ -d /home/user ] || { echo "creating /home/user (needs sudo once)"; sudo mkdir -p /home/user && sudo chown "$USER" /home/user; }
if [ -n "$DATA_DIR" ]; then
  # The code expects /home/user/data_raw; point it at the chosen folder with a symlink.
  mkdir -p "$DATA_DIR" || die "cannot create $DATA_DIR (is the drive mounted? Windows E: is /mnt/e in WSL)"
  if [ -L "$DATA" ]; then
    [ "$(readlink -f "$DATA")" = "$(readlink -f "$DATA_DIR")" ] || ln -sfn "$DATA_DIR" "$DATA"
  elif [ -e "$DATA" ]; then
    [ -z "$(ls -A "$DATA")" ] && rmdir "$DATA" && ln -s "$DATA_DIR" "$DATA" || \
      die "$DATA already exists with files; move them into $DATA_DIR and delete $DATA, or run without --data"
  else
    ln -s "$DATA_DIR" "$DATA"
  fi
  echo "data folder: $DATA -> $(readlink -f "$DATA")"
fi
mkdir -p "$DATA" "$MODELS" "$LOGS"
for t in git make unzip curl; do command -v $t >/dev/null || die "missing '$t' (sudo apt install git make unzip curl)"; done

# ---------------------------------------------------------------- 1. Python environment
setup_env() {
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  mkdir -p "$(dirname "$VENV")"
  [ -x "$VENV/bin/python" ] || uv venv -p 3.13 "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  if [ "$DEVICE" = cpu ]; then
    uv pip install -r "$REPO/requirements-lock.txt" --extra-index-url https://download.pytorch.org/whl/cpu
  else
    grep -v '^torch' "$REPO/requirements-lock.txt" > "$LOGS/req-no-torch.txt"
    uv pip install -r "$LOGS/req-no-torch.txt"
    python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || \
      uv pip install torch --index-url https://download.pytorch.org/whl/cu128
  fi
  uv pip install gdown
}
if [ $SKIP_SETUP = 0 ]; then step setup setup_env; fi
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [ $SKIP_SETUP = 1 ]; then
  echo "WARNING: no $VENV; using the current python3 ($(command -v python3)). It needs the versions in requirements-lock.txt."
else
  die "no Python environment at $VENV"
fi
export PATH="$HOME/.local/bin:$PATH"

if [ "$DEVICE" != cpu ]; then
  say "GPU check"
  python3 - <<'EOF' || die "CUDA is not usable from torch (see LOCAL_GPU.md, GPU section)"
import torch, sys
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0), "| archs:", torch.cuda.get_arch_list())
sys.exit(0 if ok else 1)
EOF
fi

# ---------------------------------------------------------------- 2. dataset
get_data() {
  local free_gb; free_gb=$(df -Pk "$(readlink -f "$DATA")" | awk 'NR==2 {print int($4/1048576)}')
  [ "$free_gb" -ge 22 ] || die "only ${free_gb} GB free in $DATA; need about 22 GB for the download + unzip"
  cd "$DATA"
  [ -f abc2027_uwb_train_v1.zip ] || gdown "$DRIVE_ID" -O abc2027_uwb_train_v1.zip
  unzip -q abc2027_uwb_train_v1.zip && rm abc2027_uwb_train_v1.zip
  cd "$REPO"
}
if [ ! -f "$PKG/streams/train/stream_12.h5" ]; then step dataset get_data; else echo "dataset already in $PKG"; fi

cd "$REPO"
MK=(make -f "$REPO/Makefile" REPO="$REPO" DEVICE="$DEVICE")
PYENV=(env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
       UWBFALL_DEVICE="$DEVICE" ABC_ROOT="$PKG" PYTHONPATH="$REPO:$PKG/src")

# ---------------------------------------------------------------- 3. features
step features "${MK[@]}" features

# ---------------------------------------------------------------- 4. train on train -> score val (official CLI)
VAL_BUNDLE=$MODELS/final_train13_$DEVICE
VAL_CSV=$DATA/submission/val_final_train13_$DEVICE.csv
mkdir -p "$DATA/submission"
train_val() {
  [ -f "$VAL_BUNDLE/meta.json" ] || "${PYENV[@]}" python3 -m uwbfall train --config "$REPO/uwbfall/configs/final.json" \
      --train-splits train --out "$VAL_BUNDLE"
  "${PYENV[@]}" python3 -m uwbfall predict --models "$VAL_BUNDLE" --split val --out "$VAL_CSV"
  "${PYENV[@]}" python3 -m uwbfall score --pred "$VAL_CSV" --split val --out "${VAL_CSV%.csv}.json"
  if [ "$DEVICE" = cpu ]; then
    if cmp -s "$VAL_CSV" "$REPO/work/I1_clip/preds/uwb_default/G7N4/val.csv"; then
      echo "val CSV byte-identical to the audited result (0.9524)"
    else
      echo "NOTE: val CSV differs from the audited CSV (different CPU/library build?). Check that the score is about 0.95."
    fi
  fi
}
step train_val train_val

# ---------------------------------------------------------------- 5. final model on train + val
step train_final "${MK[@]}" train-final
step verify "${MK[@]}" verify-models

# ---------------------------------------------------------------- 6. checks and release-day rehearsal
step checks "${MK[@]}" checks
step dry_run "${MK[@]}" dry-run

# ---------------------------------------------------------------- 7. optional SR-CV
if [ $RUN_SRCV = 1 ]; then
  run_srcv() {
    "${PYENV[@]}" python3 -m srcv.build      # fold CSVs under /home/user/data_raw/srcv (no radar data written)
    "${PYENV[@]}" python3 -m srcv.run_pipeline "local_$DEVICE" --config "$REPO/uwbfall/configs/final.json"
  }
  step srcv run_srcv
fi

# ---------------------------------------------------------------- summary
say "SUMMARY (device: $DEVICE)"
printf '%s\n' "${SUMMARY[@]}"
echo
echo "Validation score (model trained on the 13 train streams, official evaluator):"
python3 - "${VAL_CSV%.csv}.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
p = r["primary"]
print(f"  ranking F1 = {r['ranking_score']['f1_mean']:.4f}   (expected ~0.952 on CPU; GPU within about +-0.02)")
print(f"  TP/FP/FN @0.3 = {p['tp']}/{p['fp']}/{p['fn']}   FA/h = {p['false_alarms_per_hour']:.2f}   "
      f"median start error = {p['start_error_median_abs']:.3f} s")
EOF
echo
echo "Final model (train+val):  $MODELS/final"
echo "Dry-run submission:       $DATA/fake_test/submission/submission.csv"
echo "Logs:                     $LOGS"
echo "Release day:              make predict-test DEVICE=$DEVICE TEST_DIR=/path/to/streams/test   (see SUBMISSION.md)"

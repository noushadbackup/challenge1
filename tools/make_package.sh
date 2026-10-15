#!/usr/bin/env bash
# Build the one-time code submission for the organisers: dist/uwbfall_<variant>_code.zip
#
#   make package                         # VARIANT=final_norep (default)
#   make package VARIANT=final_strict
#
# Contents: the inference/training package (uwbfall/), the shipped model bundle models/<variant>/ (+ the physics-only
# fallback), Makefile, run_all.sh, requirements-lock.txt, README.md (from tools/PACKAGE_README.md), VERSION and
# SHA256SUMS. If $SUB/submission.csv exists (after release day), it is added under results/ with its logs.
# Development-only material (work/, srcv/, analysis/, reports/, parity tests) is left out; nothing in the package
# imports it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${VARIANT:-final_norep}"
SUB="${SUB:-/home/user/data_raw/submission}"
NAME="uwbfall_${VARIANT}_code"
DIST="$REPO/dist"
OUT="$DIST/$NAME"

[ -f "$REPO/models/$VARIANT/MANIFEST.sha256" ] || { echo "no shipped bundle models/$VARIANT"; exit 1; }
[ -f "$REPO/uwbfall/configs/$VARIANT.json" ] || { echo "no config uwbfall/configs/$VARIANT.json"; exit 1; }

rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/uwbfall/configs" "$OUT/uwbfall/tests" "$OUT/models"

# code: runtime modules, all configs, and the checks used by the Makefile (no parity/smoke tests: they need work/)
cp "$REPO"/uwbfall/*.py "$REPO/uwbfall/pyproject.toml" "$OUT/uwbfall/"
cp "$REPO"/uwbfall/configs/*.json "$OUT/uwbfall/configs/"
for t in __init__.py verify_bundle.py check_configs.py check_drift.py check_short.py check_submission.py; do
  cp "$REPO/uwbfall/tests/$t" "$OUT/uwbfall/tests/"
done
cp "$REPO/Makefile" "$REPO/run_all.sh" "$REPO/requirements-lock.txt" "$OUT/"
# measured scores (CPU, official evaluator): val = trained on the 13 train streams; SR-CV = pooled, TCN bag 0
case "$VARIANT" in
  final)        VAL="0.952 (64/2/3 @0.3)"; SRCV="0.918" ;;
  final_norep)  VAL="0.937 (63/3/4 @0.3)"; SRCV="0.914" ;;
  final_strict) VAL="0.937";               SRCV="0.896" ;;
  *)            VAL="see reports";         SRCV="see reports" ;;
esac
sed -e "s/{{VARIANT}}/$VARIANT/g" -e "s|{{VAL}}|$VAL|g" -e "s|{{SRCV}}|$SRCV|g" "$REPO/tools/PACKAGE_README.md" > "$OUT/README.md"

# models: the submitted bundle and the fallback, checked against their manifests first
for m in "$VARIANT" fallback_physics; do
  cp -r "$REPO/models/$m" "$OUT/models/"
  (cd "$OUT/models/$m" && sha256sum -c --quiet MANIFEST.sha256) || { echo "models/$m fails its manifest"; exit 1; }
done

# results (only once the real submission exists)
if [ -f "$SUB/submission.csv" ]; then
  mkdir -p "$OUT/results"
  for f in submission.csv submission_drift.json predict.log; do
    [ -f "$SUB/$f" ] && cp "$SUB/$f" "$OUT/results/"
  done
  echo "included results/ from $SUB"
fi

{
  echo "variant: $VARIANT"
  echo "git: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)$(git -C "$REPO" diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/VERSION"

find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
(cd "$OUT" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs sha256sum > SHA256SUMS)

python3 - "$DIST" "$NAME" <<'EOF'
import os, sys, zipfile
dist, name = sys.argv[1], sys.argv[2]
root = os.path.join(dist, name)
with zipfile.ZipFile(os.path.join(dist, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, dist))
EOF
rm -rf "$OUT"
echo "PACKAGE: $OUT.zip ($(du -h "$OUT.zip" | cut -f1))"
(cd "$DIST" && sha256sum "$NAME.zip")

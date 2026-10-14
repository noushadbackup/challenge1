"""Score / promote I2 variants with the R2_EVAL tooling (srcv.promote, official evaluator; val via official CLI).

    python3 work/I2_boundaries/score.py table V1,V2 [bags]
    python3 work/I2_boundaries/score.py promote A B [bags]
REF = I1 preds G7N4 (current reference, symlinked).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "R2_EVAL"))
import score_all as SA  # noqa: E402

HERE = Path(__file__).resolve().parent
SA.PRED = HERE / "preds"
SA.OUT = HERE / "out"
if __name__ == "__main__":
    SA.main()

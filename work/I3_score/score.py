"""Score / promote I3 variants with the R2_EVAL tooling (srcv.promote, official evaluator; val via official CLI).

    python3 work/I3_score/score.py table V1,V2 [bags]
    python3 work/I3_score/score.py promote A B [bags]
REF = reference G7N4 rebuilt by run.py (byte-identical to work/I1_clip/preds/<bag>/G7N4, checked).
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

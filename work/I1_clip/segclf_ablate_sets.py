"""Feature sets of the segment classifier (shared by segclf_ablate.py and segclf_final.py)."""


def sets(feats):
    fp = [f for f in feats if f.startswith("fpm")]
    rep = [f for f in feats if f.startswith("rep_")]
    return {"all": feats, "nofp": [f for f in feats if f not in fp], "norep": [f for f in feats if f not in rep],
            "phys": [f for f in feats if f not in fp and f not in rep],
            "phys+sp3": [f for f in feats if f not in fp and (f not in rep or f == "rep_sp3")]}

"""SR-CV: subject-disjoint regenerated cross-validation shaped like the official ABC 2027 test.

Public API (see srcv/README.md):
    folds()                        -> list of 4 fold dicts (test subjects, seeds, generated stream stats)
    sources(fold, part)            -> list[Source] virtual streams ('train' | 'test') rebuilt on demand
    real_sources(split)            -> list[Source] for the published package streams ('train' | 'val' | 'test')
    events(fold, part)             -> DataFrame, official streams_events.csv format (+ subject)
    composition(fold, part)        -> DataFrame, composition rows (+ exact times, flags, subject)
    durations(fold, part)          -> {stream_id: seconds}
    load_cir(source), load_feats(source)
    score(pred, fold)              -> official abc2027.evaluate report on the fold's test streams
    score_pooled(preds_by_fold)    -> official report on all 4 folds' test streams together (pooled OOF)
    score_val(pred)                -> official report on the real val streams (h5 durations, --ignore-flagged)
    promote(A, B, ...)             -> PLAN v2 promotion rule (paired subject-cluster bootstrap, per-fold deltas)
    bootstrap_ci(preds)            -> pooled ranking F1 + subject-cluster bootstrap 95% CI
    pipeline_sources(fold, part)   -> list[uwbfall.VirtualSource] (exact, full-precision segments)
    spec(fold, part)               -> 'comp:<csv>:<events.csv>@cv{k}_{part}_' for uwbfall.parse_sources / CLI
"""
import os as _os
import sys as _sys

_sys.dont_write_bytecode = True
_os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from .core import (  # noqa: E402,F401
    DATA, FEATS, Source, cache_feats, composition, durations, events, events_csv, fold_subjects, folds, load_cir,
    load_feats, real_events, real_sources, sources, subject_of, subjects_table,
)
from .scoring import bootstrap_ci, cluster_counts, promote, score, score_pooled, score_val  # noqa: E402,F401
from .bridge import pipeline_csv, pipeline_sources, seed_uwbfall_cache, spec  # noqa: E402,F401

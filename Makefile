# ABC 2027 UWB fall challenge: release-day / reproduction targets (see SUBMISSION.md).
#
#   make features        fill the feature cache (train, val, and test when present)
#   make train-final     retrain the $(VARIANT) bundle from scratch on train+val (18 streams) -> $(FINAL)
#   make verify-models   the retrained bundle must equal the shipped one (sha256, CPU) -> reproducibility check
#   make predict-test    one command on release day: test streams -> $(SUB)/submission.csv with the shipped $(BUNDLE)
#   make reproduce-val   $(VARIANT) trained on the 13 train streams -> val CSV -> official score
#   make package         self-contained code + model zip for the organisers (dist/), see tools/make_package.sh
#   make srcv            clean SR-CV run of final.json (4 folds + official val) and the final table vs C3
#   make dry-run         rehearsal on val copied as test_00..04 (make fake-test builds the fixture)
#   make checks          config invariants + drift-monitor sanity;  make predict-test-fallback  physics-only switch
#
# Variables (override on the command line, e.g. make predict-test TEST_DIR=/path/to/streams/test):
# REPO = the directory of this Makefile, so a clone elsewhere runs its own code (audit F1).
REPO     ?= $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
DATA     ?= /home/user/data_raw
PKG      ?= $(DATA)/abc2027_uwb_train_v1
TEST_DIR ?= $(PKG)/streams/test
MODELS   ?= $(DATA)/uwbfall_models
# Model variant (rule compliance after the organisers' Q2 answer, see reports/COMPLIANCE.md):
#   final        FINAL as audited (segment classifier uses within-stream sample-and-hold repeat statistics; replay dedups)
#   final_norep  no repeat-statistic features (replay still on distinct frames)        SR-CV -0.004
#   final_strict no repeat-statistic features and replay on all frames (no dedup)      SR-CV -0.023
VARIANT  ?= final_norep
# BUNDLE = the model predict-test uses: the shipped, sha256-verified copy in models/ (bit-reproducible on CPU).
# FINAL  = where train-final writes a from-scratch retrain; verify-models compares it with BUNDLE.
BUNDLE   ?= $(REPO)/models/$(VARIANT)
FINAL    ?= $(MODELS)/$(VARIANT)
FALLBACK ?= $(REPO)/models/fallback_physics
SUB      ?= $(DATA)/submission
CONFIG   ?= $(REPO)/uwbfall/configs/$(VARIANT).json
FB_CONFIG?= $(REPO)/uwbfall/configs/fallback_physics.json
SRCV_NAME?= final_b0
# TCN device: cpu (default; bit-exact with the shipped bundles and their manifests) | cuda | auto.
# With DEVICE=cuda a retrained bundle is not bit-identical, so verify-models only checks files and the config.
DEVICE   ?= cpu
# bash + pipefail: a failing predict piped into tee must stop make.
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c
RELAXED   = $(if $(filter cpu,$(DEVICE)),,--relaxed)
# Byte-level reference for reproduce-val (development checkout only; absent in the submission package).
VAL_REF  ?= $(if $(filter final,$(VARIANT)),$(REPO)/work/I1_clip/preds/uwb_default/G7N4/val.csv,)
PY        = PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
            UWBFALL_DEVICE=$(DEVICE) UWBFALL_CACHE=$(DATA)/uwbfall_cache ABC_ROOT=$(PKG) PYTHONPATH=$(REPO):$(PKG)/src python3

.PHONY: features train-final verify-models verify-bundle verify-fallback predict-test check-test predict-test-fallback \
        train-fallback reproduce-val srcv srcv-table checks dry-run fake-test package

features:       ## warm the feature cache (content-hashed; safe to re-run)
	$(PY) -m uwbfall extract --sources train,val
	@if ls $(TEST_DIR)/stream_*.h5 >/dev/null 2>&1; then $(PY) -m uwbfall extract --sources dir:$(TEST_DIR):test; fi

train-final:    ## ~20 min (2 threads): bundle on all 18 labelled streams; op. point from its own CV OOF
	$(PY) -m uwbfall train --config $(CONFIG) --train-splits train,val --out $(FINAL)

verify-models:  ## retrained $(FINAL) vs the shipped models/$(VARIANT)/MANIFEST.sha256 (a refit may differ only in meta timings/git rev)
	$(PY) -m uwbfall.tests.verify_bundle $(FINAL) $(REPO)/models/$(VARIANT) $(RELAXED)

verify-bundle:  ## integrity of the bundle predict-test uses ($(BUNDLE)) against the shipped manifest
	$(PY) -m uwbfall.tests.verify_bundle $(BUNDLE) $(REPO)/models/$(VARIANT) $(RELAXED)

predict-test: verify-bundle   ## release day: predict + official validate_predictions + drift monitors + checks
	mkdir -p $(SUB)
	rm -f $(SUB)/submission.csv $(SUB)/submission_drift.json
	$(PY) -m uwbfall predict --models $(BUNDLE) --sources dir:$(TEST_DIR):test --out $(SUB)/submission.csv \
	    --candidates $(SUB)/candidates.csv 2>&1 | tee $(SUB)/predict.log
	$(MAKE) -f $(REPO)/Makefile REPO=$(REPO) check-test

check-test:     ## hard checks on $(SUB)/submission.csv (ids test_00..test_04, header, official validator)
	$(PY) -m uwbfall.tests.check_submission $(SUB)/submission.csv $(TEST_DIR) test_00,test_01,test_02,test_03,test_04

verify-fallback: ## sha256 of the fallback bundle vs models/fallback_physics/MANIFEST.sha256 (refit-tolerant for meta.json)
	$(PY) -m uwbfall.tests.verify_bundle $(FALLBACK) $(REPO)/models/fallback_physics

train-fallback: ## physics-only fallback bundle (switch; only if drift monitors recommend it)
	$(PY) -m uwbfall train --config $(FB_CONFIG) --train-splits train,val --out $(MODELS)/fallback_physics

predict-test-fallback: verify-fallback
	mkdir -p $(SUB)
	rm -f $(SUB)/submission_fallback.csv
	$(PY) -m uwbfall predict --models $(FALLBACK) --sources dir:$(TEST_DIR):test --out $(SUB)/submission_fallback.csv \
	    2>&1 | tee $(SUB)/predict_fallback.log
	$(PY) -m uwbfall.tests.check_submission $(SUB)/submission_fallback.csv $(TEST_DIR) test_00,test_01,test_02,test_03,test_04

T13      ?= $(MODELS)/$(VARIANT)_train13
reproduce-val:  ## $(VARIANT) trained on the 13 train streams -> val (official CLI). Re-uses the bundle if present.
	mkdir -p $(SUB)
	@test -f $(T13)/meta.json || \
	    $(PY) -m uwbfall train --config $(CONFIG) --train-splits train --out $(T13)
	$(PY) -m uwbfall predict --models $(T13) --split val --out $(SUB)/val_$(VARIANT)_train13.csv
	$(PY) -m uwbfall score --pred $(SUB)/val_$(VARIANT)_train13.csv --split val --out $(SUB)/val_$(VARIANT)_train13.json
	@if [ -n "$(VAL_REF)" ] && [ -f "$(VAL_REF)" ]; then \
	    cmp $(SUB)/val_$(VARIANT)_train13.csv $(VAL_REF) && echo "val CSV byte-identical to the audited reference"; \
	fi

srcv:           ## ~75 min: clean SR-CV run (4 folds, then official train -> val), then the table (development checkout only)
	@test -d $(REPO)/srcv || { echo "srcv/ is not part of this package (development checkout only)"; exit 1; }
	$(PY) -m srcv.run_pipeline $(SRCV_NAME) --config $(CONFIG)
	$(PY) -m srcv.run_pipeline $(SRCV_NAME) --config $(CONFIG) --val-only
	$(MAKE) -f $(REPO)/Makefile REPO=$(REPO) srcv-table

srcv-table:     ## final table + promotion rule C3 -> FINAL (3 TCN bags), official evaluator (development checkout only)
	@test -f $(REPO)/work/FINAL/score.py || { echo "work/FINAL/score.py is not part of this package"; exit 1; }
	$(PY) $(REPO)/work/FINAL/score.py link
	$(PY) $(REPO)/work/FINAL/score.py check
	$(PY) $(REPO)/work/FINAL/score.py table
	$(PY) $(REPO)/work/FINAL/score.py promote

checks:         ## config invariants, drift-monitor sanity, short remainder streams (FIXES.md X1)
	$(PY) -m uwbfall.tests.check_configs
	$(PY) -m uwbfall.tests.check_drift $(BUNDLE) val
	$(PY) -m uwbfall.tests.check_short $(BUNDLE)

FAKE ?= $(DATA)/fake_test
FAKE_SUB ?= $(FAKE)/submission
fake-test:      ## (re)build the dry-run fixture: val streams copied as test_00..04 (1.6 GB; package untouched)
	mkdir -p $(FAKE)/streams/test
	for f in $(PKG)/streams/val/stream_*.h5; do test -f $(FAKE)/streams/test/$$(basename $$f) || cp $$f $(FAKE)/streams/test/; done
	ln -sfn $(PKG)/src $(FAKE)/src
	tr -d '\r' < $(PKG)/streams_events.csv | sed 's/^val_/test_/; s/,val_\([0-9][0-9]\)_/,test_\1_/' | awk -F, 'NR==1 || $$1 ~ /^test_/' > $(FAKE)/streams_events_fake.csv

dry-run: fake-test   ## release-day rehearsal: val streams copied as test_00..04 under $(FAKE) (output: $(FAKE_SUB))
	$(MAKE) -f $(REPO)/Makefile REPO=$(REPO) predict-test TEST_DIR=$(FAKE)/streams/test SUB=$(FAKE_SUB)

package:        ## dist/uwbfall_<VARIANT>_code.zip: code + shipped model + runbook (+ $(SUB)/submission.csv if present)
	VARIANT=$(VARIANT) SUB=$(SUB) bash $(REPO)/tools/make_package.sh

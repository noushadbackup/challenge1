# ABC 2027 UWB fall challenge: release-day / reproduction targets (see SUBMISSION.md).
#
#   make features        fill the feature cache (train, val, and test when present)
#   make train-final     FINAL bundle: configs/final.json on train+val (18 streams) -> $(FINAL)
#   make predict-test    one command on release day: test streams -> $(SUB)/submission.csv (+ drift report, checks)
#   make reproduce-val   train13 bundle with final.json -> val CSV -> official score (expect 0.9524, 64/2/3 @0.3)
#   make srcv            clean SR-CV run of final.json (4 folds + official val) and the final table vs C3
#   make dry-run         rehearsal on val copied as test_00..04 (make fake-test builds the fixture)
#   make checks          config invariants + drift-monitor sanity;  make predict-test-fallback  physics-only switch
#
# Variables (override on the command line, e.g. make predict-test TEST_DIR=/path/to/streams/test):
# REPO = the directory of this Makefile, so a clone elsewhere runs its own code (audit F1).
REPO     ?= $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PKG      ?= /home/user/data_raw/abc2027_uwb_train_v1
TEST_DIR ?= $(PKG)/streams/test
MODELS   ?= /home/user/data_raw/uwbfall_models
FINAL    ?= $(MODELS)/final
FALLBACK ?= $(MODELS)/fallback_physics
SUB      ?= /home/user/data_raw/submission
CONFIG   ?= $(REPO)/uwbfall/configs/final.json
FB_CONFIG?= $(REPO)/uwbfall/configs/fallback_physics.json
SRCV_NAME?= final_b0
PY        = PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
            ABC_ROOT=$(PKG) PYTHONPATH=$(REPO):$(PKG)/src python3

.PHONY: features train-final verify-models verify-fallback predict-test check-test predict-test-fallback train-fallback \
        reproduce-val srcv srcv-table checks dry-run fake-test

features:       ## warm the feature cache (content-hashed; safe to re-run)
	$(PY) -m uwbfall extract --sources train,val
	@if ls $(TEST_DIR)/stream_*.h5 >/dev/null 2>&1; then $(PY) -m uwbfall extract --sources dir:$(TEST_DIR):test; fi

train-final:    ## ~20 min (2 threads): FINAL bundle on all 18 labelled streams; op. point from its own CV OOF
	$(PY) -m uwbfall train --config $(CONFIG) --train-splits train,val --out $(FINAL)

verify-models:  ## sha256 of the FINAL bundle vs models/final/MANIFEST.sha256 (a refit may differ only in meta timings/git rev)
	$(PY) -m uwbfall.tests.verify_bundle $(FINAL) $(REPO)/models/final

predict-test: verify-models   ## release day: predict + official validate_predictions + drift monitors + checks
	mkdir -p $(SUB)
	$(PY) -m uwbfall predict --models $(FINAL) --sources dir:$(TEST_DIR):test --out $(SUB)/submission.csv \
	    --candidates $(SUB)/candidates.csv 2>&1 | tee $(SUB)/predict.log
	$(MAKE) -f $(REPO)/Makefile REPO=$(REPO) check-test

check-test:     ## hard checks on $(SUB)/submission.csv (ids test_00..test_04, header, official validator)
	$(PY) -m uwbfall.tests.check_submission $(SUB)/submission.csv $(TEST_DIR) test_00,test_01,test_02,test_03,test_04

verify-fallback: ## sha256 of the fallback bundle vs models/fallback_physics/MANIFEST.sha256 (refit-tolerant for meta.json)
	$(PY) -m uwbfall.tests.verify_bundle $(FALLBACK) $(REPO)/models/fallback_physics

train-fallback: ## physics-only fallback bundle (switch; only if drift monitors recommend it)
	$(PY) -m uwbfall train --config $(FB_CONFIG) --train-splits train,val --out $(FALLBACK)

predict-test-fallback: verify-fallback
	mkdir -p $(SUB)
	$(PY) -m uwbfall predict --models $(FALLBACK) --sources dir:$(TEST_DIR):test --out $(SUB)/submission_fallback.csv \
	    2>&1 | tee $(SUB)/predict_fallback.log
	$(PY) -m uwbfall.tests.check_submission $(SUB)/submission_fallback.csv $(TEST_DIR) test_00,test_01,test_02,test_03,test_04

reproduce-val:  ## final.json trained on the 13 train streams -> val (official CLI). Re-uses the bundle if present.
	@test -f $(MODELS)/final_train13/meta.json || \
	    $(PY) -m uwbfall train --config $(CONFIG) --train-splits train --out $(MODELS)/final_train13
	$(PY) -m uwbfall predict --models $(MODELS)/final_train13 --split val --out $(SUB)/val_final_train13.csv
	$(PY) -m uwbfall score --pred $(SUB)/val_final_train13.csv --split val --out $(SUB)/val_final_train13.json
	cmp $(SUB)/val_final_train13.csv $(REPO)/work/I1_clip/preds/uwb_default/G7N4/val.csv && \
	    echo "val CSV byte-identical to the I1 G7N4 bag-0 result (0.9524)"

srcv:           ## ~75 min: clean SR-CV run of final.json (4 folds, then official train -> val), then the table
	$(PY) -m srcv.run_pipeline $(SRCV_NAME) --config $(CONFIG)
	$(PY) -m srcv.run_pipeline $(SRCV_NAME) --config $(CONFIG) --val-only
	$(MAKE) -f $(REPO)/Makefile REPO=$(REPO) srcv-table

srcv-table:     ## final table + promotion rule C3 -> FINAL (3 TCN bags), official evaluator
	$(PY) $(REPO)/work/FINAL/score.py link
	$(PY) $(REPO)/work/FINAL/score.py check
	$(PY) $(REPO)/work/FINAL/score.py table
	$(PY) $(REPO)/work/FINAL/score.py promote

checks:         ## config invariants, drift-monitor sanity, submission checker on val
	$(PY) -m uwbfall.tests.check_configs
	$(PY) -m uwbfall.tests.check_drift $(FINAL) val

FAKE ?= /home/user/data_raw/fake_test
FAKE_SUB ?= $(FAKE)/submission
fake-test:      ## (re)build the dry-run fixture: val streams copied as test_00..04 (1.6 GB; package untouched)
	mkdir -p $(FAKE)/streams/test
	for f in $(PKG)/streams/val/stream_*.h5; do test -f $(FAKE)/streams/test/$$(basename $$f) || cp $$f $(FAKE)/streams/test/; done
	ln -sfn $(PKG)/src $(FAKE)/src
	tr -d '\r' < $(PKG)/streams_events.csv | sed 's/^val_/test_/; s/,val_\([0-9][0-9]\)_/,test_\1_/' | awk -F, 'NR==1 || $$1 ~ /^test_/' > $(FAKE)/streams_events_fake.csv

dry-run: fake-test   ## release-day rehearsal: val streams copied as test_00..04 under $(FAKE) (output: $(FAKE_SUB))
	$(MAKE) -f $(REPO)/Makefile REPO=$(REPO) predict-test TEST_DIR=$(FAKE)/streams/test SUB=$(FAKE_SUB)

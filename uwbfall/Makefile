# uwbfall shortcuts (run from anywhere: make -C /home/user/challenge1/uwbfall <target>)
REPO   ?= /home/user/challenge1
MODELS ?= /home/user/data_raw/uwbfall_models
RUNS   ?= /home/user/data_raw/uwbfall_runs
CONFIG ?= $(REPO)/uwbfall/configs/reference.json
NAME   ?= default_train13
SPLITS ?= train
PY      = PYTHONPATH=$(REPO) PYTHONDONTWRITEBYTECODE=1 python3

.PHONY: extract train retune predict-val score-val predict-test parity verify-repro

extract:        ## fill the feature cache for train, val (and test when present)
	$(PY) -m uwbfall extract --sources train,val

train:          ## fit a bundle
	$(PY) -m uwbfall train --config $(CONFIG) --train-splits $(SPLITS) --out $(MODELS)/$(NAME)

retune:         ## upgrade bundle $(FROM) to $(CONFIG)'s clip/decode/boundary stages without refitting HGB/TCN
	$(PY) -m uwbfall retune --models $(MODELS)/$(FROM) --config $(CONFIG) --train-splits $(SPLITS) --out $(MODELS)/$(NAME)

predict-val:
	$(PY) -m uwbfall predict --models $(MODELS)/$(NAME) --split val --out $(RUNS)/val_$(NAME).csv

score-val:
	$(PY) -m uwbfall score --pred $(RUNS)/val_$(NAME).csv --split val

predict-test:   ## needs streams/test/stream_XX.h5
	$(PY) -m uwbfall predict --models $(MODELS)/$(NAME) --split test --out $(RUNS)/test_$(NAME).csv

parity:         ## replica checks: extractor/design vs team caches, writer vs published h5, regenerated bank
	$(PY) -m uwbfall.tests.parity_features
	$(PY) -m uwbfall.tests.parity_regenerated 1001 4
	$(PY) -m uwbfall.tests.parity_virtual val 4

verify-repro:   ## train on 13 train streams with the INTEGRATION config, predict + score val, compare artefacts
	$(MAKE) train CONFIG=$(REPO)/uwbfall/configs/integration_repro.json NAME=repro_train13
	$(MAKE) predict-val NAME=repro_train13
	$(MAKE) score-val NAME=repro_train13
	$(PY) -m uwbfall predict --models $(MODELS)/repro_train13 --split val --out $(RUNS)/val_repro_train13.csv --candidates $(RUNS)/val_repro_train13_cand.csv
	$(PY) -m uwbfall.tests.parity_models $(MODELS)/repro_train13 $(RUNS)/val_repro_train13.csv $(RUNS)/val_repro_train13_cand.csv
	cmp $(RUNS)/val_repro_train13.csv $(REPO)/work/INTEGRATION/out/val_tcn75_proto25_replayB.csv && echo "val CSV byte-identical to INTEGRATION"

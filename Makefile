.PHONY: all paper-results experiments robustness temporal acceptance temporal-acceptance temporal-fetch temporal-all verify-inputs compile test check manifest

PYTHON ?= python3

all: paper-results

paper-results:
	$(PYTHON) scripts/run_all_experiments.py \
		--with-robustness \
		--with-temporal \
		--with-acceptance

experiments:
	$(PYTHON) scripts/run_all_experiments.py --skip-latex

robustness:
	$(PYTHON) scripts/run_all_experiments.py --with-robustness

temporal:
	$(PYTHON) scripts/validate_temporal.py

acceptance temporal-acceptance:
	$(PYTHON) scripts/validate_acceptance_analyses.py \
		--cohort data/processed/temporal_cohort_20210617.csv \
		--dv data/processed/temporal_dv_20210617.csv \
		--xml data/processed/temporal_xml_20210617.csv \
		--outdir reports/acceptance

temporal-fetch:
	$(PYTHON) scripts/fetch_temporal_inputs.py
	$(PYTHON) scripts/build_temporal_cohort.py
	$(PYTHON) scripts/fetch_temporal_dv.py
	$(PYTHON) scripts/fetch_temporal_xml.py

temporal-all: temporal-fetch temporal acceptance

verify-inputs:
	$(PYTHON) scripts/verify_inputs.py

compile:
	$(PYTHON) -m compileall -q scripts tests

test:
	$(PYTHON) -m unittest discover -s tests -v

check: compile test
	$(PYTHON) scripts/run_all_experiments.py \
		--with-robustness \
		--with-temporal \
		--with-acceptance \
		--robustness-seeds 2 \
		--bootstrap-replicates 10 \
		--temporal-seeds 2 \
		--temporal-bootstrap-replicates 10 \
		--acceptance-bootstrap-replicates 10 \
		--dry-run \
		--manifest /tmp/tess-dv-tcv-dry-run-manifest.json

manifest:
	$(PYTHON) scripts/run_all_experiments.py --skip-latex

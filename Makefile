PY ?= .venv/bin/python

.PHONY: install test status reproduce figures

install:
	python -m venv .venv
	$(PY) -m pip install -r requirements.txt pytest

test:
	$(PY) -m pytest tests -q

status:
	$(PY) -m btcregime status

reproduce:
	$(PY) src/baseline.py > /dev/null
	$(PY) src/audit.py > /dev/null
	$(PY) src/exp10_map.py
	$(PY) src/exp10_bayes.py
	$(PY) src/exp12_vol_predict.py
	$(PY) src/holdout.py > /dev/null
	$(PY) src/exp17_robustness.py

figures:
	$(PY) src/paper_figures.py

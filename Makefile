PY := .venv/bin/python
# macOS has no OpenMP runtime unless Homebrew's libomp is installed, and XGBoost needs one.
# scikit-learn ships a copy; point the loader at it. (Harmless on Linux.)
export DYLD_FALLBACK_LIBRARY_PATH := $(shell $(PY) -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))" 2>/dev/null)

.PHONY: venv data sweep promote report ui test

venv:
	python3.12 -m venv .venv
	$(PY) -m pip install -r requirements.txt

data:
	mkdir -p data/raw
	curl -fsSL -o data/raw/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
	curl -fsSL -o data/raw/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

sweep:
	$(PY) -m exptrack.sweep

promote:
	$(PY) -m exptrack.registry

report:
	$(PY) -m exptrack.report

ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

test:
	$(PY) -m pytest

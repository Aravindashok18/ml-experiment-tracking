import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exptrack.data import CATEGORICAL, NUMERIC, TARGET  # noqa: E402
from exptrack.validate import ALLOWED  # noqa: E402

COUNTRIES = ["United-States", "Mexico", "India", "Germany", "Philippines"]


def make_adult_like(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """A frame with Adult's columns and a learnable target (a logistic function of a few columns)."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "education-num": rng.integers(1, 17, n).astype(float),
        "capital-gain": np.where(rng.random(n) < 0.08, rng.integers(500, 20000, n), 0).astype(float),
        "capital-loss": np.where(rng.random(n) < 0.05, rng.integers(100, 2000, n), 0).astype(float),
        "hours-per-week": rng.integers(10, 70, n).astype(float),
    })
    for column in CATEGORICAL:
        choices = sorted(ALLOWED[column]) if column in ALLOWED else COUNTRIES
        frame[column] = pd.Series(rng.choice(choices, n), dtype=object)
    for column in ("workclass", "occupation", "native-country"):  # a few missing values, like the real data
        frame.loc[rng.random(n) < 0.04, column] = np.nan
    logit = (0.07 * (frame["age"] - 38) + 0.35 * (frame["education-num"] - 10) + 0.04 * (frame["hours-per-week"] - 40)
             + 0.0002 * frame["capital-gain"] + 0.8 * (frame["sex"] == "Male") + 1.5 * (frame["marital-status"] == "Married-civ-spouse") - 2.2)
    frame[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int64")
    return frame


@pytest.fixture()
def adult_like():
    return make_adult_like


@pytest.fixture()
def train_test():
    return make_adult_like(1500, seed=1), make_adult_like(600, seed=2)


@pytest.fixture()
def store(tmp_path):
    """A fresh MLflow tracking store (SQLite file + artifact folder) for one test."""
    import mlflow

    from exptrack import tracking

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracking.setup(uri, artifact_root=str(tmp_path / "artifacts"))
    yield uri
    mlflow.set_tracking_uri("sqlite:///:memory:")


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:  # an XGBoostError, not ImportError, when the OpenMP runtime is missing
        return False


needs_xgboost = pytest.mark.skipif(not xgboost_available(), reason="xgboost cannot load (macOS needs the OpenMP runtime, see the README)")

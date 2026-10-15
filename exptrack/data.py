"""Loading and cleaning the UCI Adult census-income data (the ETL step).

The data comes as two headerless CSV files with the official split: adult.data (train) and
adult.test (held out). adult.test has a junk first line and writes its labels with a trailing
dot ("<=50K."), and both mark missing values with "?".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"  # 1 when income is above 50K


@dataclass
class CleaningReport:
    rows_read: int
    duplicates_removed: int
    rows_out: int
    missing_cells: Dict[str, int] = field(default_factory=dict)
    positive_rate: float = 0.0


def load_raw(path: Path, is_test: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(
        path, header=None, names=RAW_COLUMNS, skipinitialspace=True, na_values="?",
        skiprows=1 if is_test else 0, skip_blank_lines=True,
    )
    return frame.dropna(how="all").reset_index(drop=True)


def clean(raw: pd.DataFrame, drop_duplicates: bool) -> Tuple[pd.DataFrame, CleaningReport]:
    """Turn a raw frame into model-ready columns.

    - label: ">50K" (with or without the test file's trailing dot) becomes target 1, "<=50K" becomes 0
    - fnlwgt (a survey sampling weight) and education (a duplicate of education-num) are dropped
    - numeric columns become float64 so a missing value never changes a column's type
    - exact duplicate rows are removed from the training data only; the held-out file stays as published
    """
    frame = raw.copy()
    rows_read = len(frame)
    income = frame["income"].astype("string").str.strip().str.rstrip(".")
    unexpected = set(income.dropna().unique()) - {"<=50K", ">50K"}
    if unexpected or income.isna().any():
        raise ValueError(f"unexpected income labels: {sorted(unexpected)} (missing: {int(income.isna().sum())})")
    frame[TARGET] = (income == ">50K").astype("int64")

    duplicates = 0
    if drop_duplicates:
        before = len(frame)
        frame = frame.drop_duplicates().reset_index(drop=True)
        duplicates = before - len(frame)

    frame = frame[FEATURES + [TARGET]].copy()
    frame[NUMERIC] = frame[NUMERIC].astype("float64")
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("string").str.strip().astype(object)
        frame.loc[frame[column].isna(), column] = np.nan

    report = CleaningReport(
        rows_read=rows_read,
        duplicates_removed=duplicates,
        rows_out=len(frame),
        missing_cells={c: int(frame[c].isna().sum()) for c in FEATURES if frame[c].isna().any()},
        positive_rate=float(frame[TARGET].mean()),
    )
    return frame, report


def load_adult(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, CleaningReport]]:
    train, train_report = clean(load_raw(data_dir / "adult.data"), drop_duplicates=True)
    test, test_report = clean(load_raw(data_dir / "adult.test", is_test=True), drop_duplicates=False)
    return train, test, {"train": train_report, "test": test_report}


def fingerprint(frame: pd.DataFrame) -> str:
    """Stable SHA-256 of a frame's contents (values, column names and order), for tagging runs with their data."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def split_xy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    return frame[FEATURES], frame[TARGET].to_numpy()

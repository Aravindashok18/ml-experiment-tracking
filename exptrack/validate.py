"""Data-validation checks that run before any model is trained.

Each check returns a pass/fail with a detail string, and the whole report is saved as JSON and
logged with every MLflow run, so a result can always be traced to data that passed these checks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET

# Allowed values come from the dataset's own documentation (adult.names).
ALLOWED = {
    "workclass": {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"},
    "marital-status": {"Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"},
    "occupation": {"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                   "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"},
    "relationship": {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"},
    "race": {"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"},
    "sex": {"Female", "Male"},
}
# native-country has 41 values; it is checked for shape (no stray whitespace or numbers) rather than listed here.
RANGES = {
    "age": (17, 100), "education-num": (1, 16), "capital-gain": (0, 100_000),
    "capital-loss": (0, 5_000), "hours-per-week": (1, 99),
}
MAX_MISSING_FRACTION = 0.10  # per column; Adult's worst is occupation at about 6%
MIN_ROWS = 1_000
PSI_WARN = 0.25  # the usual rule of thumb: above 0.25 the distributions differ materially


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    dataset: str
    rows: int
    checks: List[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    def to_json(self) -> str:
        return json.dumps({"dataset": self.dataset, "rows": self.rows, "passed": self.passed,
                           "checks": [asdict(c) for c in self.checks]}, indent=2)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json())


class ValidationError(Exception):
    pass


def validate(frame: pd.DataFrame, dataset: str) -> ValidationReport:
    checks: List[Check] = []

    def add(name: str, passed: bool, detail: str = "ok") -> None:
        checks.append(Check(name, bool(passed), detail))

    expected = set(FEATURES + [TARGET])
    missing_cols, extra_cols = expected - set(frame.columns), set(frame.columns) - expected
    add("columns", not missing_cols and not extra_cols, f"missing={sorted(missing_cols)} extra={sorted(extra_cols)}")
    if missing_cols:  # nothing else can be checked sensibly
        return ValidationReport(dataset, len(frame), checks)

    add("row_count", len(frame) >= MIN_ROWS, f"{len(frame)} rows (minimum {MIN_ROWS})")
    add("target_values", set(frame[TARGET].unique()) <= {0, 1}, f"values={sorted(frame[TARGET].unique().tolist())}")
    add("both_classes_present", frame[TARGET].nunique() == 2, f"positive rate {frame[TARGET].mean():.3f}")

    for column in NUMERIC:
        values = frame[column]
        add(f"{column}:numeric", pd.api.types.is_float_dtype(values), f"dtype={values.dtype}")
        add(f"{column}:no_missing", values.notna().all(), f"{int(values.isna().sum())} missing")
        low, high = RANGES[column]
        outside = int(((values < low) | (values > high)).sum())
        add(f"{column}:range", outside == 0, f"{outside} values outside [{low}, {high}]")

    for column in CATEGORICAL:
        values = frame[column]
        missing_fraction = float(values.isna().mean())
        add(f"{column}:missing_rate", missing_fraction <= MAX_MISSING_FRACTION, f"{missing_fraction:.1%} (limit {MAX_MISSING_FRACTION:.0%})")
        present = set(values.dropna().unique())
        if column in ALLOWED:
            unknown = sorted(present - ALLOWED[column])
            add(f"{column}:allowed_values", not unknown, f"unknown={unknown}")
        else:
            odd = sorted(v for v in present if not isinstance(v, str) or v != v.strip() or any(ch.isdigit() for ch in v))
            add(f"{column}:well_formed", not odd, f"odd={odd[:5]}")
    return ValidationReport(dataset, len(frame), checks)


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples of one numeric column, using the expected sample's decile edges.

    PSI = sum((actual% - expected%) * ln(actual% / expected%)). Zero means identical distributions.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # a (near-)constant column: compare the share equal to the mode instead
        edges = np.array([expected.min() - 1, np.median(expected), expected.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compare_train_test(train: pd.DataFrame, test: pd.DataFrame) -> ValidationReport:
    """Is the held-out data drawn from the same population as the training data?"""
    checks: List[Check] = []
    for column in NUMERIC:
        psi = population_stability_index(train[column].to_numpy(), test[column].to_numpy())
        checks.append(Check(f"{column}:psi", psi < PSI_WARN, f"PSI {psi:.4f} (limit {PSI_WARN})"))
    for column in CATEGORICAL:
        unseen = sorted(set(test[column].dropna().unique()) - set(train[column].dropna().unique()))
        checks.append(Check(f"{column}:unseen_in_train", not unseen, f"unseen={unseen}"))
    drift = abs(float(test[TARGET].mean()) - float(train[TARGET].mean()))
    checks.append(Check("positive_rate_gap", drift < 0.03, f"train {train[TARGET].mean():.3f} vs test {test[TARGET].mean():.3f}"))
    return ValidationReport("train_vs_test", len(test), checks)


def require(*reports: ValidationReport) -> None:
    failures = [(r.dataset, c) for r in reports for c in r.failures()]
    if failures:
        lines = "\n".join(f"  [{d}] {c.name}: {c.detail}" for d, c in failures)
        raise ValidationError(f"data validation failed:\n{lines}")

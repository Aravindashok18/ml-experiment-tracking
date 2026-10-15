"""Model families, their hyperparameter search spaces, and the seeded sampler that draws configurations."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC

FAMILIES = ("logistic_regression", "random_forest", "xgboost")


@dataclass(frozen=True)
class Config:
    family: str
    params: Dict[str, Any]

    @property
    def config_id(self) -> str:
        """Short stable id of the family and hyperparameters, used to skip runs that already exist."""
        blob = json.dumps({"family": self.family, "params": self.params}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def make_estimator(family: str, params: Dict[str, Any], seed: int = 0):
    if family == "logistic_regression":
        # liblinear supports both penalties and is fast on a dataset this size
        return LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed, **params)
    if family == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    if family == "xgboost":
        from xgboost import XGBClassifier  # imported late: on macOS it needs an OpenMP runtime (see the README)

        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=seed, eval_metric="logloss", **params)
    if family == "baseline":
        return DummyClassifier(strategy="prior")
    raise ValueError(f"unknown model family {family!r}")


def build_pipeline(family: str, params: Dict[str, Any], seed: int = 0) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", make_estimator(family, normalise_params(family, params), seed))])


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.round(10 ** rng.uniform(np.log10(low), np.log10(high)), 5))


def _sample(family: str, rng: np.random.Generator) -> Dict[str, Any]:
    if family == "logistic_regression":
        return {
            "C": _log_uniform(rng, 1e-3, 1e2),
            "penalty": str(rng.choice(["l1", "l2"])),
            "class_weight": None if rng.random() < 0.5 else "balanced",
        }
    if family == "random_forest":
        depth = rng.choice([6, 10, 16, 0])
        return {
            "n_estimators": int(rng.choice([100, 200, 300])),
            "max_depth": None if depth == 0 else int(depth),
            "min_samples_leaf": int(rng.choice([1, 3, 5, 10])),
            "max_features": str(rng.choice(["sqrt", "log2", "0.3"])),
            "class_weight": None if rng.random() < 0.5 else "balanced_subsample",
        }
    if family == "xgboost":
        return {
            "n_estimators": int(rng.choice([100, 200, 400])),
            "max_depth": int(rng.choice([3, 4, 6, 8])),
            "learning_rate": _log_uniform(rng, 0.02, 0.3),
            "subsample": float(np.round(rng.uniform(0.6, 1.0), 3)),
            "colsample_bytree": float(np.round(rng.uniform(0.5, 1.0), 3)),
            "min_child_weight": int(rng.choice([1, 3, 5])),
            "reg_lambda": _log_uniform(rng, 0.1, 10),
        }
    raise ValueError(family)


def normalise_params(family: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sampled values to what scikit-learn expects (max_features "0.3" -> 0.3)."""
    fixed = dict(params)
    if family == "random_forest" and isinstance(fixed.get("max_features"), str):
        try:
            fixed["max_features"] = float(fixed["max_features"])
        except ValueError:
            pass
    return fixed


def generate_configs(counts: Dict[str, int], seed: int = 0) -> List[Config]:
    """Draw `counts[family]` distinct random configurations per family, reproducibly."""
    configs: List[Config] = []
    for index, family in enumerate(FAMILIES):
        rng = np.random.default_rng([seed, index])
        seen, drawn, attempts = set(), 0, 0
        while drawn < counts.get(family, 0):
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(f"could not draw {counts[family]} distinct {family} configurations")
            config = Config(family, _sample(family, rng))
            if config.config_id in seen:
                continue
            seen.add(config.config_id)
            configs.append(config)
            drawn += 1
    return configs

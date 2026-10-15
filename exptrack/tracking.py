"""One experiment run: cross-validate a configuration, evaluate it on the held-out test set,
and record everything in MLflow so the run can be compared, reproduced and promoted later."""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from .data import FEATURES, fingerprint, split_xy  # noqa: E402
from .models import Config, build_pipeline  # noqa: E402
from .validate import ValidationReport  # noqa: E402

EXPERIMENT = "adult-income"
CV_SPLITS = 5
CV_SCORING = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}


def setup(tracking_uri: str, experiment: str = EXPERIMENT, artifact_root: Optional[str] = None) -> str:
    """Point MLflow at a tracking store, make `experiment` the active one (creating it if needed), return its id."""
    mlflow.set_tracking_uri(tracking_uri)
    found = mlflow.get_experiment_by_name(experiment)
    experiment_id = found.experiment_id if found else mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment_name=experiment)  # without this, runs land in the "Default" experiment
    return experiment_id


def existing_config_ids(experiment: str = EXPERIMENT) -> set:
    """Config ids that already have a finished run, so an interrupted sweep can resume."""
    runs = mlflow.search_runs(experiment_names=[experiment], filter_string="attributes.status = 'FINISHED'", output_format="pandas")
    column = "tags.config_id"
    return set(runs[column].dropna()) if len(runs) and column in runs else set()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _confusion_figure(y_true, y_pred, title: str):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], ["<=50K", ">50K"])
    ax.set_yticks([0, 1], ["<=50K", ">50K"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def _feature_table(pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "importance"
    elif hasattr(model, "coef_"):
        values, label = model.coef_.ravel(), "coefficient"
    else:
        return pd.DataFrame({"feature": [], "value": []})
    table = pd.DataFrame({"feature": names, label: values})
    return table.reindex(table[label].abs().sort_values(ascending=False).index).head(25)


def run_experiment(
    config: Config,
    train: pd.DataFrame,
    test: pd.DataFrame,
    reports: Dict[str, ValidationReport],
    seed: int = 0,
    stage: str = "sweep",
    log_model: bool = True,
) -> Dict[str, Any]:
    """Cross-validate and evaluate one configuration inside a new MLflow run. Returns its metrics and run id."""
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    pipeline = build_pipeline(config.family, config.params, seed)
    folds = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)  # same folds for every run

    with mlflow.start_run(run_name=f"{config.family}-{config.config_id}") as run:
        mlflow.set_tags({
            "config_id": config.config_id, "family": config.family, "stage": stage,
            "git_commit": _git_commit(), "dataset": "uci-adult",
            "train_sha256": fingerprint(train)[:16], "test_sha256": fingerprint(test)[:16],
        })
        mlflow.log_params({f"hp.{k}": ("None" if v is None else v) for k, v in config.params.items()})
        mlflow.log_params({"seed": seed, "cv_splits": CV_SPLITS, "train_rows": len(train), "test_rows": len(test),
                           "n_features_in": len(FEATURES)})

        # 1. cross-validation on the training data: this is what models are selected on
        cv = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=CV_SCORING, n_jobs=1)
        metrics: Dict[str, float] = {}
        for name in CV_SCORING:
            scores = cv[f"test_{name}"]
            metrics[f"cv_{name}_mean"], metrics[f"cv_{name}_std"] = float(scores.mean()), float(scores.std(ddof=1))
            for fold, value in enumerate(scores):
                metrics[f"cv_{name}_fold{fold}"] = float(value)
        metrics["cv_fit_seconds_mean"] = float(cv["fit_time"].mean())

        # 2. final fit on all training data, one evaluation on the held-out test set
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        metrics["train_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["predict_seconds_per_1k_rows"] = (time.perf_counter() - started) / len(X_test) * 1000
        predicted = (proba >= 0.5).astype(int)
        metrics.update({
            "test_accuracy": accuracy_score(y_test, predicted),
            "test_f1": f1_score(y_test, predicted),
            "test_precision": precision_score(y_test, predicted, zero_division=0),
            "test_recall": recall_score(y_test, predicted),
            "test_roc_auc": roc_auc_score(y_test, proba),
            "test_pr_auc": average_precision_score(y_test, proba),
        })
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        # 3. artifacts: data checks, predictions (for paired comparisons later), plots, feature effects, the model
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, report in reports.items():
                report.save(tmp / f"validation_{name}.json")
            np.save(tmp / "test_proba.npy", proba.astype("float32"))
            _feature_table(pipeline).to_csv(tmp / "top_features.csv", index=False)
            fig = _confusion_figure(y_test, predicted, f"{config.family} (test, threshold 0.5)")
            fig.savefig(tmp / "confusion_matrix.png", dpi=120)
            plt.close(fig)
            mlflow.log_artifacts(str(tmp), artifact_path="diagnostics")
        if log_model:
            sample = X_test.head(50)
            # cloudpickle, not MLflow 3's skops default: skops refuses numpy dtypes and XGBoost classes unless every
            # type is allowlisted. Only load models from a tracking store you trust.
            mlflow.sklearn.log_model(
                pipeline, name="model", signature=infer_signature(sample, pipeline.predict(sample)),
                input_example=sample.head(3), serialization_format="cloudpickle",
            )
        return {"run_id": run.info.run_id, "config_id": config.config_id, "family": config.family, **metrics}

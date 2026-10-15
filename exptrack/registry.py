"""Promote the best run's model to the MLflow Model Registry, behind a gate.

    python -m exptrack.registry            # register the best run if it passes the gate

Selection uses cross-validated ROC AUC on the training data only. The held-out test score is
a sanity check on the choice, never an input to it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow import MlflowClient

from . import tracking

MODEL_NAME = "adult-income-classifier"
CHAMPION = "champion"
SELECT_ON = "cv_roc_auc_mean"
MIN_TEST_AUC = 0.85       # an absolute floor on the held-out score
MAX_CV_TEST_GAP = 0.02    # |cv AUC - test AUC| beyond this suggests leakage or a mismatched split
MIN_IMPROVEMENT = 0.0005  # a new champion must beat the current one by at least this much CV AUC


def best_run(experiment: str = tracking.EXPERIMENT, metric: str = SELECT_ON):
    runs = mlflow.search_runs(
        experiment_names=[experiment], filter_string="attributes.status = 'FINISHED' and tags.family != 'baseline'",
        order_by=[f"metrics.{metric} DESC"], max_results=1, output_format="pandas",
    )
    if runs.empty:
        raise RuntimeError("no finished runs to promote; run `python -m exptrack.sweep` first")
    return runs.iloc[0]


def _champion_cv_auc(client: MlflowClient, name: str) -> Optional[float]:
    try:
        version = client.get_model_version_by_alias(name, CHAMPION)
    except mlflow.exceptions.MlflowException:
        return None
    return float(client.get_run(version.run_id).data.metrics[SELECT_ON])


def promote(experiment: str = tracking.EXPERIMENT, name: str = MODEL_NAME) -> Dict[str, Any]:
    client = MlflowClient()
    run = best_run(experiment)
    cv_auc, test_auc = float(run[f"metrics.{SELECT_ON}"]), float(run["metrics.test_roc_auc"])
    info = {"run_id": run["run_id"], "family": run["tags.family"], "cv_roc_auc": cv_auc, "test_roc_auc": test_auc}

    if test_auc < MIN_TEST_AUC:
        return {**info, "promoted": False, "reason": f"test AUC {test_auc:.4f} is below the floor of {MIN_TEST_AUC}"}
    if abs(cv_auc - test_auc) > MAX_CV_TEST_GAP:
        return {**info, "promoted": False, "reason": f"CV AUC {cv_auc:.4f} and test AUC {test_auc:.4f} differ by more than {MAX_CV_TEST_GAP}"}
    current = _champion_cv_auc(client, name)
    if current is not None and cv_auc < current + MIN_IMPROVEMENT:
        return {**info, "promoted": False, "reason": f"CV AUC {cv_auc:.4f} does not beat the current champion's {current:.4f} by {MIN_IMPROVEMENT}"}

    version = mlflow.register_model(f"runs:/{run['run_id']}/model", name)
    client.update_model_version(name, version.version, description=(
        f"{info['family']} chosen on cross-validated ROC AUC {cv_auc:.4f}; held-out test ROC AUC {test_auc:.4f}."))
    for key, value in {"family": info["family"], "run_id": info["run_id"], "cv_roc_auc": f"{cv_auc:.4f}",
                       "test_roc_auc": f"{test_auc:.4f}", "train_sha256": run["tags.train_sha256"]}.items():
        client.set_model_version_tag(name, version.version, key, value)
    client.set_registered_model_alias(name, CHAMPION, version.version)
    return {**info, "promoted": True, "version": int(version.version), "replaced_cv_roc_auc": current}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tracking-uri", default=f"sqlite:///{root / 'mlflow.db'}")
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    result = promote()
    if result["promoted"]:
        print(f"registered {MODEL_NAME} v{result['version']} ({result['family']}, cv AUC {result['cv_roc_auc']:.4f}, "
              f"test AUC {result['test_roc_auc']:.4f}) and moved the '{CHAMPION}' alias to it")
    else:
        print(f"not promoted: {result['reason']}")


if __name__ == "__main__":
    main()

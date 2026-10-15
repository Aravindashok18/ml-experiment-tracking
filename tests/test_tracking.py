import json

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from conftest import needs_xgboost
from exptrack import tracking
from exptrack.models import Config
from exptrack.validate import compare_train_test, validate

RF = Config("random_forest", {"n_estimators": 8, "max_depth": None, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": None})


@pytest.fixture()
def reports(train_test):
    train, test = train_test
    return {"train": validate(train, "train"), "test": validate(test, "test"), "train_vs_test": compare_train_test(train, test)}


def run_one(store, train_test, reports, config=RF, **kwargs):
    train, test = train_test
    return tracking.run_experiment(config, train, test, reports, **kwargs)


def test_a_run_records_tags_params_and_metrics(store, train_test, reports):
    result = run_one(store, train_test, reports)
    run = MlflowClient().get_run(result["run_id"])

    assert run.info.status == "FINISHED"
    assert run.data.tags["family"] == "random_forest" and run.data.tags["config_id"] == RF.config_id
    assert run.data.tags["stage"] == "sweep" and run.data.tags["dataset"] == "uci-adult" and run.data.tags["git_commit"]
    assert len(run.data.tags["train_sha256"]) == 16
    assert run.data.params["hp.n_estimators"] == "8" and run.data.params["hp.max_depth"] == "None"
    assert run.data.params["cv_splits"] == "5" and run.data.params["seed"] == "0"
    for metric in ("cv_roc_auc_mean", "cv_roc_auc_std", "cv_f1_mean", "cv_accuracy_mean", "test_roc_auc", "test_f1", "test_pr_auc",
                   "test_precision", "test_recall", "test_accuracy", "train_seconds", "predict_seconds_per_1k_rows"):
        assert metric in run.data.metrics, metric
    assert 0.5 < run.data.metrics["test_roc_auc"] <= 1.0


def test_every_cv_fold_is_logged_and_the_mean_is_their_average(store, train_test, reports):
    result = run_one(store, train_test, reports)
    metrics = MlflowClient().get_run(result["run_id"]).data.metrics

    folds = [metrics[f"cv_roc_auc_fold{i}"] for i in range(5)]
    assert metrics["cv_roc_auc_mean"] == pytest.approx(np.mean(folds))
    assert metrics["cv_roc_auc_std"] == pytest.approx(np.std(folds, ddof=1))
    assert "cv_roc_auc_fold5" not in metrics


def test_artifacts_include_checks_predictions_plots_and_a_loadable_model(store, train_test, reports, tmp_path):
    train, test = train_test
    result = run_one(store, train_test, reports)
    client = MlflowClient()
    names = {a.path for a in client.list_artifacts(result["run_id"], "diagnostics")}

    assert names == {"diagnostics/validation_train.json", "diagnostics/validation_test.json", "diagnostics/validation_train_vs_test.json",
                     "diagnostics/test_proba.npy", "diagnostics/top_features.csv", "diagnostics/confusion_matrix.png"}
    proba = np.load(mlflow.artifacts.download_artifacts(run_id=result["run_id"], artifact_path="diagnostics/test_proba.npy"))
    assert proba.shape == (len(test),) and proba.dtype == np.float32
    saved = json.loads(open(mlflow.artifacts.download_artifacts(run_id=result["run_id"], artifact_path="diagnostics/validation_train.json")).read())
    assert saved["passed"] is True

    model = mlflow.sklearn.load_model(f"runs:/{result['run_id']}/model")
    from exptrack.data import split_xy
    X, _ = split_xy(test)
    assert model.predict_proba(X.head(10)).shape == (10, 2)
    np.testing.assert_allclose(model.predict_proba(X)[:, 1], proba, atol=1e-6)  # the saved model reproduces the saved predictions


LR = Config("logistic_regression", {"C": 1.0, "penalty": "l2", "class_weight": None})


def test_the_same_configuration_gives_the_same_scores(store, train_test, reports):
    first, second = run_one(store, train_test, reports, config=LR), run_one(store, train_test, reports, config=LR)

    assert first["run_id"] != second["run_id"]
    assert first["cv_roc_auc_mean"] == second["cv_roc_auc_mean"]
    assert first["test_roc_auc"] == second["test_roc_auc"]


def test_forest_scores_repeat_to_four_decimals_but_not_bit_for_bit(store, train_test, reports):
    # Threads add up the trees' predictions in whatever order they finish, which shifts floating-point sums by ~1e-16 and
    # reorders tied scores, so a seeded random forest repeats to about 1e-5 rather than exactly.
    first, second = run_one(store, train_test, reports), run_one(store, train_test, reports)

    assert first["cv_roc_auc_mean"] == pytest.approx(second["cv_roc_auc_mean"], abs=1e-4)
    assert first["test_roc_auc"] == pytest.approx(second["test_roc_auc"], abs=1e-4)


def test_models_can_be_left_out_to_save_space(store, train_test, reports):
    result = run_one(store, train_test, reports, log_model=False)

    assert MlflowClient().list_artifacts(result["run_id"]) and not any(a.path == "model" for a in MlflowClient().list_artifacts(result["run_id"]))


def test_the_baseline_scores_chance_and_every_model_beats_it(store, train_test, reports):
    baseline = run_one(store, train_test, reports, config=Config("baseline", {}), log_model=False)
    forest = run_one(store, train_test, reports, log_model=False)

    assert baseline["cv_roc_auc_mean"] == pytest.approx(0.5)
    assert baseline["cv_f1_mean"] == 0
    assert forest["cv_roc_auc_mean"] > baseline["cv_roc_auc_mean"] + 0.2


def test_finished_configurations_are_listed_but_failed_ones_are_not(store, train_test, reports):
    run_one(store, train_test, reports, log_model=False)
    broken = Config("random_forest", {"n_estimators": -5})
    with pytest.raises(Exception):
        run_one(store, train_test, reports, config=broken, log_model=False)

    assert tracking.existing_config_ids() == {RF.config_id}
    statuses = {r.data.tags["config_id"]: r.info.status for r in MlflowClient().search_runs([tracking.setup(store)])}
    assert statuses[broken.config_id] == "FAILED"


def test_setup_is_idempotent(store):
    assert tracking.setup(store) == tracking.setup(store)


@needs_xgboost
def test_xgboost_runs_are_tracked_like_any_other(store, train_test, reports):
    config = Config("xgboost", {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.2, "subsample": 0.9, "colsample_bytree": 0.9,
                                "min_child_weight": 1, "reg_lambda": 1.0})

    result = run_one(store, train_test, reports, config=config, log_model=False)

    assert result["family"] == "xgboost" and result["cv_roc_auc_mean"] > 0.7

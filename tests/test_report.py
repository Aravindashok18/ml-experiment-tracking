import numpy as np
import pandas as pd
import pytest

from exptrack import report, tracking
from exptrack.data import split_xy
from exptrack.models import Config
from exptrack.validate import compare_train_test, validate


def runs_frame(rows):
    base = {"family": "random_forest", "config_id": "x", "run_id": "r", "cv_roc_auc_std": 0.003, "cv_f1_mean": 0.6, "test_f1": 0.6,
            "test_accuracy": 0.8, "train_seconds": 1.0}
    return pd.DataFrame([{**base, **row} for row in rows])


# --- the statistics, on hand-made numbers -------------------------------------------------

def test_paired_bootstrap_of_identical_models_is_exactly_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = rng.random(2000)

    result = report.paired_bootstrap(y, proba, proba, "roc_auc", resamples=100)

    assert result["diff"] == 0 and result["ci_low"] == 0 and result["ci_high"] == 0 and result["share_a_better"] == 0


def test_paired_bootstrap_detects_a_clearly_better_model_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 3000)
    good = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 3000), 0, 1)
    poor = rng.random(3000)

    first = report.paired_bootstrap(y, good, poor, "roc_auc", resamples=200, seed=5)
    again = report.paired_bootstrap(y, good, poor, "roc_auc", resamples=200, seed=5)
    flipped = report.paired_bootstrap(y, poor, good, "roc_auc", resamples=200, seed=5)

    assert first == again
    assert first["ci_low"] > 0.2 and first["share_a_better"] == 1.0
    assert flipped["diff"] == pytest.approx(-first["diff"], abs=1e-9)
    f1 = report.paired_bootstrap(y, good, poor, "f1", resamples=200)
    assert f1["ci_low"] > 0


def test_f1_helper_matches_the_definition():
    y = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])

    assert report._f1(y, pred) == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert report._f1(np.zeros(4), np.zeros(4)) == 0.0


def test_hyperparameter_effects_find_a_monotonic_relationship_and_skip_tiny_samples():
    rows = [{"cv_roc_auc_mean": 0.80 + 0.01 * i, "hp_n_estimators": 100 + 10 * i, "hp_min_samples_leaf": [1, 3, 5][i % 3],
             "hp_max_features": [0.3, 0.5][i % 2]} for i in range(10)]
    runs = runs_frame(rows)

    effects = report.hyperparameter_effects(runs).set_index("hyperparameter")

    assert effects.loc["n_estimators", "spearman_rho"] == pytest.approx(1.0)
    assert effects.loc["n_estimators", "p_value"] < 0.001
    assert "min_samples_leaf" in effects.index
    assert "max_features" not in effects.index  # two distinct values is not enough for a rank correlation
    assert effects.loc["n_estimators", "significant_after_bonferroni"] and not effects.loc["min_samples_leaf", "significant_after_bonferroni"]
    assert report.hyperparameter_effects(runs.head(5)).empty


def test_champion_is_chosen_on_cross_validation_not_on_the_test_score():
    runs = runs_frame([
        {"family": "xgboost", "config_id": "a", "cv_roc_auc_mean": 0.92, "test_roc_auc": 0.90},
        {"family": "xgboost", "config_id": "b", "cv_roc_auc_mean": 0.91, "test_roc_auc": 0.95},  # best on test, but not chosen
    ])

    summary = report.family_summary(runs)

    assert summary.loc[0, "champion_test_auc"] == 0.90 and summary.loc[0, "cv_auc_best"] == 0.92


def test_cv_and_test_rankings_agree_perfectly_when_runs_are_ordered_the_same_way():
    runs = runs_frame([{"cv_roc_auc_mean": 0.8 + 0.01 * i, "test_roc_auc": 0.7 + 0.01 * i} for i in range(8)])

    rho, _ = report.cv_vs_test_agreement(runs)

    assert rho == pytest.approx(1.0)


def test_markdown_table_formats_numbers_and_leaves_text_alone():
    frame = pd.DataFrame({"name": ["a"], "value": [0.123456]})

    assert report._md(frame, {"value": ".2f"}).splitlines() == ["| name | value |", "|---|---|", "| a | 0.12 |"]


# --- end to end on a small real store -------------------------------------------------------

def test_summary_and_plots_come_out_of_a_tracked_experiment(store, train_test, tmp_path):
    train, test = train_test
    reports = {"train": validate(train, "train"), "test": validate(test, "test"), "train_vs_test": compare_train_test(train, test)}
    configs = [Config("baseline", {})] + [
        Config("logistic_regression", {"C": c, "penalty": "l2", "class_weight": None}) for c in (0.01, 0.1, 1.0, 10.0, 0.5, 2.0, 5.0, 0.05, 20.0)
    ] + [
        Config("random_forest", {"n_estimators": n, "max_depth": d, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": None})
        for n, d in ((5, 3), (8, 4), (12, 5), (6, 6), (10, 8), (7, 10), (9, 12), (11, 14), (13, 16))
    ]
    for config in configs:
        tracking.run_experiment(config, train, test, reports, log_model=False)

    runs = report.load_runs()
    _, y_test = split_xy(test)
    summary = report.build_summary(runs, y_test, resamples=50)
    report.plot_auc_by_family(runs[runs["family"] != "baseline"], tmp_path / "a.png")
    report.plot_auc_vs_time(runs[runs["family"] != "baseline"], tmp_path / "b.png")

    assert len(runs) == 19 and runs["hp_C"].notna().sum() == 9 and runs["hp_max_depth"].notna().sum() == 9
    assert "# Experiment summary" in summary and "19 finished runs (18 model configurations" in summary
    assert "Logistic Regression" in summary and "Random Forest" in summary and "Paired bootstrap" in summary
    assert "| Logistic Regression | Random Forest | test ROC AUC |" in summary or "test ROC AUC" in summary
    assert (tmp_path / "a.png").stat().st_size > 1000 and (tmp_path / "b.png").stat().st_size > 1000

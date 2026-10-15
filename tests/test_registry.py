import mlflow
import pytest
from mlflow import MlflowClient

from exptrack import registry, tracking
from exptrack.data import split_xy
from exptrack.models import Config
from exptrack.validate import compare_train_test, validate

GOOD = Config("logistic_regression", {"C": 1.0, "penalty": "l2", "class_weight": None})
# a stump: strong regularisation alone would not make a logistic regression weak, since AUC only needs the ranking
WEAK = Config("random_forest", {"n_estimators": 2, "max_depth": 1, "min_samples_leaf": 50, "max_features": "sqrt", "class_weight": None})
BETTER = Config("random_forest", {"n_estimators": 60, "max_depth": 8, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": None})


@pytest.fixture()
def tracked(store, train_test, monkeypatch):
    train, test = train_test
    reports = {"train": validate(train, "train"), "test": validate(test, "test"), "train_vs_test": compare_train_test(train, test)}
    # the synthetic data is not Adult, and a deliberately weak model has a noisy CV/test gap: relax the gates here, test them below
    monkeypatch.setattr(registry, "MIN_TEST_AUC", 0.5)
    monkeypatch.setattr(registry, "MAX_CV_TEST_GAP", 1.0)

    def run(config):
        return tracking.run_experiment(config, train, test, reports)

    return run


def test_the_best_run_by_cv_auc_is_registered_and_gets_the_champion_alias(tracked):
    tracked(Config("baseline", {}))
    tracked(WEAK)
    good = tracked(GOOD)

    result = registry.promote()
    client = MlflowClient()
    version = client.get_model_version_by_alias(registry.MODEL_NAME, registry.CHAMPION)

    assert result["promoted"] and result["version"] == 1 and result["run_id"] == good["run_id"]
    assert int(version.version) == 1 and version.run_id == good["run_id"]
    assert version.tags["family"] == "logistic_regression" and version.tags["cv_roc_auc"] == f"{good['cv_roc_auc_mean']:.4f}"
    assert version.tags["train_sha256"] and "cross-validated" in version.description


def test_the_baseline_is_never_promoted(tracked):
    tracked(Config("baseline", {}))

    with pytest.raises(RuntimeError, match="no finished runs"):
        registry.promote()


def test_running_promote_again_without_a_better_model_adds_no_version(tracked):
    tracked(GOOD)
    registry.promote()

    second = registry.promote()

    assert not second["promoted"] and "does not beat the current champion" in second["reason"]
    assert len(MlflowClient().search_model_versions(f"name='{registry.MODEL_NAME}'")) == 1


def test_a_better_model_becomes_version_two_and_takes_the_alias(tracked):
    tracked(WEAK)
    registry.promote()
    better = tracked(BETTER)

    result = registry.promote()
    client = MlflowClient()

    assert result["promoted"] and result["version"] == 2 and result["replaced_cv_roc_auc"] < better["cv_roc_auc_mean"]
    assert int(client.get_model_version_by_alias(registry.MODEL_NAME, registry.CHAMPION).version) == 2
    assert len(client.search_model_versions(f"name='{registry.MODEL_NAME}'")) == 2  # version 1 is kept


def test_the_champion_can_be_loaded_by_alias_and_predicts(tracked, train_test):
    tracked(GOOD)
    registry.promote()
    _, test = train_test
    X, _ = split_xy(test)

    model = mlflow.sklearn.load_model(f"models:/{registry.MODEL_NAME}@{registry.CHAMPION}")

    assert model.predict_proba(X.head(7)).shape == (7, 2)


def test_a_test_score_below_the_floor_blocks_promotion(tracked, monkeypatch):
    tracked(GOOD)
    monkeypatch.setattr(registry, "MIN_TEST_AUC", 0.999)

    result = registry.promote()

    assert not result["promoted"] and "below the floor" in result["reason"]
    assert MlflowClient().search_registered_models() == []


def test_a_large_gap_between_cv_and_test_blocks_promotion(tracked, monkeypatch):
    tracked(GOOD)
    monkeypatch.setattr(registry, "MAX_CV_TEST_GAP", 0.0)

    result = registry.promote()

    assert not result["promoted"] and "differ by more than" in result["reason"]

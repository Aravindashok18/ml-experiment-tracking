import pytest

from conftest import needs_xgboost
from exptrack.data import split_xy
from exptrack.models import FAMILIES, Config, build_pipeline, generate_configs, normalise_params

COUNTS = {"logistic_regression": 12, "random_forest": 18, "xgboost": 18}


def test_the_sampler_draws_the_requested_number_per_family():
    configs = generate_configs(COUNTS, seed=0)

    assert len(configs) == 48
    for family, count in COUNTS.items():
        assert sum(c.family == family for c in configs) == count


def test_configurations_are_unique_and_reproducible():
    first, second = generate_configs(COUNTS, seed=0), generate_configs(COUNTS, seed=0)

    assert [c.config_id for c in first] == [c.config_id for c in second]
    assert len({c.config_id for c in first}) == len(first)
    assert [c.config_id for c in generate_configs(COUNTS, seed=1)] != [c.config_id for c in first]


def test_adding_runs_to_one_family_does_not_change_the_others():
    small = {c.config_id for c in generate_configs({"xgboost": 5}, seed=0)}
    large = {c.config_id for c in generate_configs({"xgboost": 18}, seed=0)}

    assert small <= large  # a bigger sweep extends a smaller one, so finished runs are never redone


def test_sampled_values_stay_inside_the_search_space():
    for config in generate_configs(COUNTS, seed=3):
        p = config.params
        if config.family == "logistic_regression":
            assert 1e-3 <= p["C"] <= 1e2 and p["penalty"] in ("l1", "l2") and p["class_weight"] in (None, "balanced")
        elif config.family == "random_forest":
            assert p["n_estimators"] in (100, 200, 300) and p["max_depth"] in (6, 10, 16, None)
            assert p["min_samples_leaf"] in (1, 3, 5, 10) and p["max_features"] in ("sqrt", "log2", "0.3")
        else:
            assert 0.02 <= p["learning_rate"] <= 0.3 and 0.6 <= p["subsample"] <= 1.0 and 0.5 <= p["colsample_bytree"] <= 1.0
            assert p["max_depth"] in (3, 4, 6, 8) and p["n_estimators"] in (100, 200, 400) and 0.1 <= p["reg_lambda"] <= 10


def test_config_id_depends_on_family_and_values_only():
    a = Config("random_forest", {"n_estimators": 100, "max_depth": None})
    assert a.config_id == Config("random_forest", {"max_depth": None, "n_estimators": 100}).config_id
    assert a.config_id != Config("random_forest", {"n_estimators": 200, "max_depth": None}).config_id
    assert a.config_id != Config("xgboost", {"n_estimators": 100, "max_depth": None}).config_id


def test_a_fraction_given_as_text_becomes_a_number_for_scikit_learn():
    assert normalise_params("random_forest", {"max_features": "0.3"})["max_features"] == 0.3
    assert normalise_params("random_forest", {"max_features": "sqrt"})["max_features"] == "sqrt"
    assert normalise_params("logistic_regression", {"C": 1.0}) == {"C": 1.0}


@pytest.mark.parametrize("family,params", [
    ("baseline", {}),
    ("logistic_regression", {"C": 0.5, "penalty": "l1", "class_weight": "balanced"}),
    ("random_forest", {"n_estimators": 10, "max_depth": 4, "min_samples_leaf": 3, "max_features": "0.3", "class_weight": "balanced_subsample"}),
    pytest.param("xgboost", {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.2}, marks=needs_xgboost),
])
def test_every_family_fits_and_predicts_probabilities_on_data_with_missing_values(adult_like, family, params):
    frame = adult_like(400)
    X, y = split_xy(frame)

    pipeline = build_pipeline(family, params).fit(X, y)
    proba = pipeline.predict_proba(X.head(25))

    assert proba.shape == (25, 2) and ((proba >= 0) & (proba <= 1)).all()


def test_an_unseen_category_at_prediction_time_does_not_crash(adult_like):
    frame = adult_like(400)
    X, y = split_xy(frame)
    pipeline = build_pipeline("logistic_regression", {"C": 1.0}).fit(X, y)
    X = X.head(5).copy()
    X.loc[X.index[0], "occupation"] = "Brand-new-job"

    assert pipeline.predict_proba(X).shape == (5, 2)


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown model family"):
        build_pipeline("svm", {})


def test_the_three_families_are_the_ones_the_sweep_runs():
    assert set(FAMILIES) == {"logistic_regression", "random_forest", "xgboost"}

import numpy as np
import pandas as pd
import pytest

from exptrack.data import TARGET
from exptrack.validate import (
    PSI_WARN, ValidationError, compare_train_test, population_stability_index, require, validate,
)


def failed(report):
    return {c.name for c in report.failures()}


def test_clean_data_passes_every_check(adult_like):
    report = validate(adult_like(2000), "train")

    assert report.passed and len(report.checks) == 33
    assert report.failures() == []


@pytest.mark.parametrize("column,value,check", [
    ("age", -5, "age:range"), ("age", 150, "age:range"), ("education-num", 0, "education-num:range"),
    ("hours-per-week", 120, "hours-per-week:range"), ("capital-gain", -1, "capital-gain:range"),
    ("workclass", "Astronaut", "workclass:allowed_values"), ("sex", "Unknown", "sex:allowed_values"),
    ("native-country", "Mexico ", "native-country:well_formed"), ("native-country", "42", "native-country:well_formed"),
    (TARGET, 2, "target_values"),
])
def test_each_kind_of_bad_value_is_caught_by_the_right_check(adult_like, column, value, check):
    frame = adult_like(2000)
    frame[column] = frame[column].astype(object) if column in ("workclass", "sex", "native-country") else frame[column]
    frame.loc[0, column] = value

    assert check in failed(validate(frame, "x"))


def test_missing_numbers_and_too_many_missing_categories_are_caught(adult_like):
    frame = adult_like(2000)
    frame.loc[:5, "age"] = np.nan
    frame.loc[: int(0.3 * len(frame)), "occupation"] = np.nan

    assert {"age:no_missing", "occupation:missing_rate"} <= failed(validate(frame, "x"))


def test_integer_columns_are_flagged_because_a_later_missing_value_would_change_their_type(adult_like):
    frame = adult_like(2000)
    frame["age"] = frame["age"].astype(int)

    assert "age:numeric" in failed(validate(frame, "x"))


def test_missing_column_extra_column_too_few_rows_and_one_class(adult_like):
    frame = adult_like(2000)

    assert "columns" in failed(validate(frame.drop(columns=["race"]), "x"))
    assert "columns" in failed(validate(frame.assign(surprise=1), "x"))
    assert "row_count" in failed(validate(frame.head(50), "x"))
    assert "both_classes_present" in failed(validate(frame.assign(**{TARGET: 0}), "x"))


def test_a_missing_column_stops_further_checks_instead_of_crashing(adult_like):
    report = validate(adult_like(2000).drop(columns=["age"]), "x")

    assert [c.name for c in report.checks] == ["columns"]


# --- drift ---------------------------------------------------------------------------------

def test_psi_is_zero_for_identical_samples_and_large_for_a_shifted_one():
    rng = np.random.default_rng(0)
    base = rng.normal(40, 10, 20_000)

    assert population_stability_index(base, base) == pytest.approx(0, abs=1e-9)
    assert population_stability_index(base, rng.normal(40, 10, 20_000)) < 0.01
    assert population_stability_index(base, rng.normal(55, 10, 20_000)) > PSI_WARN


def test_psi_copes_with_columns_that_are_mostly_one_value():
    rng = np.random.default_rng(1)
    mostly_zero = np.where(rng.random(5000) < 0.92, 0.0, rng.integers(1, 9000, 5000))

    assert np.isfinite(population_stability_index(mostly_zero, mostly_zero))
    assert np.isfinite(population_stability_index(np.zeros(100), np.ones(100)))


def test_similar_train_and_test_pass_the_drift_checks(adult_like):
    assert compare_train_test(adult_like(3000, seed=1), adult_like(3000, seed=2)).passed


def test_drift_checks_flag_a_shifted_column_an_unseen_category_and_a_label_gap(adult_like):
    train, test = adult_like(3000, seed=1), adult_like(3000, seed=2)
    test["age"] = test["age"] + 25
    test.loc[0, "workclass"] = "Never-worked-ever"
    test[TARGET] = 1

    assert {"age:psi", "workclass:unseen_in_train", "positive_rate_gap"} <= failed(compare_train_test(train, test))


def test_require_raises_with_every_failure_listed(adult_like):
    bad = adult_like(2000)
    bad.loc[0, "age"] = -1
    bad.loc[1, "sex"] = "??"

    with pytest.raises(ValidationError) as error:
        require(validate(bad, "train"), validate(adult_like(2000), "test"))

    message = str(error.value)
    assert "[train] age:range" in message and "[train] sex:allowed_values" in message and "[test]" not in message


def test_report_round_trips_to_json(adult_like, tmp_path):
    report = validate(adult_like(2000), "train")
    path = tmp_path / "r.json"
    report.save(path)

    import json

    saved = json.loads(path.read_text())
    assert saved["passed"] is True and saved["rows"] == 2000 and len(saved["checks"]) == len(report.checks)

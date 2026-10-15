import numpy as np
import pandas as pd
import pytest

from exptrack.data import FEATURES, TARGET, clean, fingerprint, load_adult, load_raw

TRAIN = """39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, ?, Not-in-family, White, Male, 0, 0, 40, ?, >50K
38, Private, 215646, HS-grad, 9, Divorced, ?, Not-in-family, White, Male, 0, 0, 40, ?, >50K

"""
TEST = """|1x3 Cross validator
25, Private, 226802, 11th, 7, Never-married, Machine-op-inspct, Own-child, Black, Male, 0, 0, 40, United-States, <=50K.
38, Private, 89814, HS-grad, 9, Married-civ-spouse, Farming-fishing, Husband, White, Male, 0, 0, 50, United-States, >50K.
38, Private, 89814, HS-grad, 9, Married-civ-spouse, Farming-fishing, Husband, White, Male, 0, 0, 50, United-States, >50K.
"""


@pytest.fixture()
def data_dir(tmp_path):
    (tmp_path / "adult.data").write_text(TRAIN)
    (tmp_path / "adult.test").write_text(TEST)
    return tmp_path


def test_test_file_header_line_is_skipped_and_question_marks_become_missing(data_dir):
    test = load_raw(data_dir / "adult.test", is_test=True)
    train = load_raw(data_dir / "adult.data")

    assert len(test) == 3 and test.iloc[0]["age"] == 25
    assert train["occupation"].isna().sum() == 2 and train["native-country"].isna().sum() == 2


def test_labels_become_zero_or_one_with_or_without_the_trailing_dot(data_dir):
    train, test, _ = load_adult(data_dir)

    assert train[TARGET].tolist() == [0, 0, 1]
    assert test[TARGET].tolist() == [0, 1, 1]


def test_duplicates_are_removed_from_train_only(data_dir):
    train, test, report = load_adult(data_dir)

    assert (report["train"].rows_read, report["train"].duplicates_removed, report["train"].rows_out) == (4, 1, 3)
    assert (report["test"].rows_read, report["test"].duplicates_removed, report["test"].rows_out) == (3, 0, 3)


def test_cleaned_frame_has_model_columns_only_and_float_numbers(data_dir):
    train, _, _ = load_adult(data_dir)

    assert list(train.columns) == FEATURES + [TARGET]
    assert "fnlwgt" not in train and "education" not in train
    assert all(train[c].dtype == "float64" for c in ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"])


def test_missing_values_are_counted(data_dir):
    _, _, report = load_adult(data_dir)

    assert report["train"].missing_cells == {"occupation": 1, "native-country": 1}
    assert report["train"].positive_rate == pytest.approx(1 / 3)


def test_unexpected_label_is_an_error_not_a_silent_zero(data_dir):
    raw = load_raw(data_dir / "adult.data")
    raw.loc[0, "income"] = "maybe"

    with pytest.raises(ValueError, match="maybe"):
        clean(raw, drop_duplicates=True)


def test_fingerprint_is_stable_and_sensitive(adult_like):
    frame = adult_like(200, seed=0)
    same = frame.copy()
    changed = frame.copy()
    changed.loc[3, "age"] += 1

    assert fingerprint(frame) == fingerprint(same)
    assert fingerprint(frame) != fingerprint(changed)
    assert fingerprint(frame) != fingerprint(frame[list(reversed(frame.columns))])
    assert len(fingerprint(frame)) == 64

import pandas as pd
import pytest

from my_library.splitters.stratified import StratifiedKFoldSplitter


# pytest my_library/tests/unit/splitters/test_stratified.py -v
@pytest.fixture
def sample_data():
    # 10 samples with balanced classes 0 and 1
    X = pd.DataFrame({'feature': range(10)})
    y = pd.Series([0, 1] * 5, name='target')
    return X, y


def test_split_return_type_and_length(sample_data):
    X, y = sample_data
    splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=0)
    folds = splitter.split(X, y)
    assert isinstance(folds, list), "split should return a list"
    assert len(folds) == 5, "should return one fold per split"

    for ((X_tr, y_tr), (X_val, y_val)) in folds:
        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_val, pd.Series)


def test_stratification_preserved(sample_data):
    X, y = sample_data
    splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=0)
    folds = splitter.split(X, y)

    # each validation fold should contain exactly one 0 and one 1
    for (_, _), (_, y_val) in folds:
        counts = y_val.value_counts().to_dict()
        assert counts.get(0, 0) == 1
        assert counts.get(1, 0) == 1


def test_total_samples_preserved(sample_data):
    X, y = sample_data
    splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=0)
    n = len(X)
    for (X_tr, y_tr), (X_val, y_val) in splitter.split(X, y):
        assert len(X_tr) + len(X_val) == n
        assert len(y_tr) + len(y_val) == n


def test_reproducible_splits(sample_data):
    X, y = sample_data
    s1 = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=123)
    s2 = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=123)
    folds1 = s1.split(X, y)
    folds2 = s2.split(X, y)

    for ((X1_tr, y1_tr), (X1_val, y1_val)), ((X2_tr, y2_tr), (X2_val, y2_val)) \
        in zip(folds1, folds2, strict=False):
        pd.testing.assert_frame_equal(X1_tr, X2_tr)
        pd.testing.assert_series_equal(y1_tr, y2_tr)
        pd.testing.assert_frame_equal(X1_val, X2_val)
        pd.testing.assert_series_equal(y1_val, y2_val)


def test_groups_ignored(sample_data):
    X, y = sample_data
    groups = pd.Series(['A'] * len(X))
    splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=0)

    folds_no_grp = splitter.split(X, y)
    folds_with_grp = splitter.split(X, y, groups=groups)

    # passing groups should not change the splits
    for ((X0_tr, y0_tr), (X0_val, y0_val)), ((X1_tr, y1_tr), (X1_val, y1_val)) \
        in zip(folds_no_grp, folds_with_grp, strict=False):
        pd.testing.assert_frame_equal(X0_tr, X1_tr)
        pd.testing.assert_series_equal(y0_tr, y1_tr)
        pd.testing.assert_frame_equal(X0_val, X1_val)
        pd.testing.assert_series_equal(y0_val, y1_val)


def test_mismatched_lengths_raises():
    # X and y of different lengths
    X = pd.DataFrame({'feature': range(9)})
    y = pd.Series([0, 1] * 5, name='target')  # length 10
    splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=0)
    with pytest.raises(ValueError):
        splitter.split(X, y)

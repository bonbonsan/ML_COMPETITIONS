import numpy as np
import pandas as pd
import pytest

from my_library.splitters.time_series import TimeSeriesSplitter


# pytest my_library/tests/unit/splitters/test_time_series.py -v
@pytest.fixture
def sample_data():
    # 10 sequential samples
    X = pd.DataFrame({'feature': np.arange(10)})
    y = pd.Series(np.arange(10), name='target')
    return X, y


def test_split_return_type_and_length_expanding(sample_data):
    X, y = sample_data
    splitter = TimeSeriesSplitter(n_splits=3, method="expanding")
    folds = splitter.split(X, y)
    # Expect 3 folds
    assert isinstance(folds, list)
    assert len(folds) == 3

    # Check each fold is tuple of train/valid pairs
    for ((X_tr, y_tr), (X_val, y_val)) in folds:
        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_val, pd.Series)


def test_fold_sizes_expanding(sample_data):
    X, y = sample_data
    splitter = TimeSeriesSplitter(n_splits=3, method="expanding")
    folds = splitter.split(X, y)
    # default test_size = 10 // (3+1) = 2
    expected = [
        (2, 2),  # fold 0: train size 2, valid size 2
        (4, 2),  # fold 1: train size 4, valid size 2
        (6, 2),  # fold 2: train size 6, valid size 2
    ]
    for ((X_tr, y_tr), (X_val, y_val)), (exp_train, exp_valid) \
        in zip(folds, expected, strict=False):
        assert len(X_tr) == exp_train
        assert len(y_tr) == exp_train
        assert len(X_val) == exp_valid
        assert len(y_val) == exp_valid


def test_split_return_type_and_length_rolling(sample_data):
    X, y = sample_data
    splitter = TimeSeriesSplitter(n_splits=3, method="rolling", max_train_size=4)
    folds = splitter.split(X, y)
    assert len(folds) == 3
    # Check each fold has expected sizes
    # default test_size = 2
    sizes = [(2,2), (4,2), (4,2)]
    for ((X_tr, _), (X_val, _)), (exp_tr, exp_val) in zip(folds, sizes, strict=False):
        assert len(X_tr) == exp_tr
        assert len(X_val) == exp_val


def test_custom_test_size_and_fewer_folds(sample_data):
    X, y = sample_data
    # override test_size=3
    splitter = TimeSeriesSplitter(n_splits=5, method="expanding", test_size=3)
    folds = splitter.split(X, y)
    # Only two folds possible: (3,3) and (6,3)
    assert len(folds) == 2
    expected = [(3,3), (6,3)]
    for ((X_tr, _), (X_val, _)), (exp_tr, exp_val) in zip(folds, expected, strict=False):
        assert len(X_tr) == exp_tr
        assert len(X_val) == exp_val


def test_groups_ignored(sample_data):
    X, y = sample_data
    groups = pd.Series(['G'] * len(X))
    splitter = TimeSeriesSplitter(n_splits=3, method="expanding")
    folds_no = splitter.split(X, y)
    folds_grp = splitter.split(X, y, groups=groups)
    # Ensure identical splits
    for ((X0_tr, y0_tr), (X0_val, y0_val)), ((X1_tr, y1_tr), (X1_val, y1_val)) \
        in zip(folds_no, folds_grp, strict=False):
        pd.testing.assert_frame_equal(X0_tr, X1_tr)
        pd.testing.assert_series_equal(y0_tr, y1_tr)
        pd.testing.assert_frame_equal(X0_val, X1_val)
        pd.testing.assert_series_equal(y0_val, y1_val)


def test_reproducible_splits(sample_data):
    X, y = sample_data
    s1 = TimeSeriesSplitter(n_splits=3, method="expanding")
    s2 = TimeSeriesSplitter(n_splits=3, method="expanding")
    folds1 = s1.split(X, y)
    folds2 = s2.split(X, y)
    for ((X1_tr, y1_tr), (X1_val, y1_val)), ((X2_tr, y2_tr), (X2_val, y2_val)) \
        in zip(folds1, folds2, strict=False):
        pd.testing.assert_frame_equal(X1_tr, X2_tr)
        pd.testing.assert_series_equal(y1_tr, y2_tr)
        pd.testing.assert_frame_equal(X1_val, X2_val)
        pd.testing.assert_series_equal(y1_val, y2_val)


def test_mismatched_lengths_raises(sample_data):
    X, y = sample_data
    # Drop one element from y to cause mismatch
    y_bad = y.iloc[:-1]
    splitter = TimeSeriesSplitter(n_splits=3, method="expanding")
    with pytest.raises(ValueError):
        splitter.split(X, y_bad)

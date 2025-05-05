import numpy as np
import pandas as pd
import pytest

from my_library.splitters.grouped import GroupKFoldSplitter


# pytest my_library/tests/unit/splitters/test_grouped.py -v
@pytest.fixture
def sample_data():
    # 12 samples, 6 groups (A–F), each repeated twice
    X = pd.DataFrame({'feature': range(12)})
    y = pd.Series(range(12), name='target')
    groups = pd.Series(np.repeat(list('ABCDEF'), 2), name='group')
    return X, y, groups


def test_split_return_type_and_length(sample_data):
    X, y, groups = sample_data
    splitter = GroupKFoldSplitter(n_splits=5)
    folds = splitter.split(X, y, groups)
    assert isinstance(folds, list)
    assert len(folds) == 5

    for ((X_tr, y_tr), (X_val, y_val)) in folds:
        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_val, pd.Series)


def test_no_group_leakage(sample_data):
    X, y, groups = sample_data
    splitter = GroupKFoldSplitter(n_splits=5)
    folds = splitter.split(X, y, groups)

    for ((X_tr, _), (X_val, _)) in folds:
        train_groups = set(groups.loc[X_tr.index])
        val_groups = set(groups.loc[X_val.index])
        assert train_groups.isdisjoint(val_groups), \
            "Train and validation sets share the same group"


def test_total_samples_preserved(sample_data):
    X, y, groups = sample_data
    splitter = GroupKFoldSplitter(n_splits=5)
    n_samples = len(X)
    for ((X_tr, y_tr), (X_val, y_val)) in splitter.split(X, y, groups):
        assert len(X_tr) + len(X_val) == n_samples
        assert len(y_tr) + len(y_val) == n_samples


def test_reproducible_splits(sample_data):
    X, y, groups = sample_data
    s1 = GroupKFoldSplitter(n_splits=5)
    s2 = GroupKFoldSplitter(n_splits=5)
    folds1 = s1.split(X, y, groups)
    folds2 = s2.split(X, y, groups)

    for ((X1_tr, y1_tr), (X1_val, y1_val)), ((X2_tr, y2_tr), (X2_val, y2_val)) \
            in zip(folds1, folds2, strict=False):
        pd.testing.assert_frame_equal(X1_tr, X2_tr)
        pd.testing.assert_series_equal(y1_tr, y2_tr)
        pd.testing.assert_frame_equal(X1_val, X2_val)
        pd.testing.assert_series_equal(y1_val, y2_val)


def test_mismatched_lengths_raises(sample_data):
    X, y, groups = sample_data
    # groups of incorrect length
    bad_groups = groups.iloc[:-1]
    splitter = GroupKFoldSplitter(n_splits=5)
    with pytest.raises(ValueError):
        splitter.split(X, y, bad_groups)

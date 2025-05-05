import numpy as np
import pandas as pd
import pytest

from my_library.splitters.stratified_grouped import StratifiedGroupKFoldSplitter


# pytest my_library/tests/unit/splitters/test_stratified_grouped.py -v
@pytest.fixture
def sample_data():
    # 12 samples, 6 groups (A–F), each group constant target: A–C→0, D–F→1
    X = pd.DataFrame({'feature': range(12)})
    groups = pd.Series(np.repeat(list('ABCDEF'), 2), name='group')
    y = pd.Series([0] * 6 + [1] * 6, name='target')
    return X, y, groups


def test_split_return_type_and_length(sample_data):
    X, y, groups = sample_data
    splitter = StratifiedGroupKFoldSplitter(n_splits=3, random_state=0)
    folds = splitter.split(X, y, groups)
    assert isinstance(folds, list)
    assert len(folds) == 3

    for ((X_tr, y_tr), (X_val, y_val)) in folds:
        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_val, pd.Series)


def test_stratification_preserved(sample_data):
    X, y, groups = sample_data
    splitter = StratifiedGroupKFoldSplitter(n_splits=3, random_state=0)
    folds = splitter.split(X, y, groups)

    # Each validation fold should contain exactly 2 samples of each class
    for ((_, _), (_, y_val)) in folds:
        counts = y_val.value_counts().to_dict()
        assert counts.get(0, 0) == 2
        assert counts.get(1, 0) == 2


def test_no_group_leakage(sample_data):
    X, y, groups = sample_data
    splitter = StratifiedGroupKFoldSplitter(n_splits=3, random_state=0)
    folds = splitter.split(X, y, groups)

    for ((X_tr, _), (X_val, _)) in folds:
        train_groups = set(groups.loc[X_tr.index])
        val_groups = set(groups.loc[X_val.index])
        assert train_groups.isdisjoint(val_groups)


def test_total_samples_preserved(sample_data):
    X, y, groups = sample_data
    splitter = StratifiedGroupKFoldSplitter(n_splits=3, random_state=0)
    total = len(X)
    for (X_tr, y_tr), (X_val, y_val) in splitter.split(X, y, groups):
        assert len(X_tr) + len(X_val) == total
        assert len(y_tr) + len(y_val) == total


def test_reproducible_splits(sample_data):
    X, y, groups = sample_data
    s1 = StratifiedGroupKFoldSplitter(n_splits=3, random_state=123)
    s2 = StratifiedGroupKFoldSplitter(n_splits=3, random_state=123)
    folds1 = s1.split(X, y, groups)
    folds2 = s2.split(X, y, groups)

    for ((X1_tr, y1_tr), (X1_val, y1_val)), ((X2_tr, y2_tr), (X2_val, y2_val)) \
        in zip(folds1, folds2, strict=False):
        pd.testing.assert_frame_equal(X1_tr, X2_tr)
        pd.testing.assert_series_equal(y1_tr, y2_tr)
        pd.testing.assert_frame_equal(X1_val, X2_val)
        pd.testing.assert_series_equal(y1_val, y2_val)


def test_mismatched_lengths_raises():
    X = pd.DataFrame({'feature': range(5)})
    y = pd.Series(range(5), name='target')
    groups = pd.Series(['A', 'A', 'B', 'B'], name='group')  # length 4 != length 5
    splitter = StratifiedGroupKFoldSplitter(n_splits=2)
    with pytest.raises(ValueError):
        splitter.split(X, y, groups)

import numpy as np
import pandas as pd
import pytest

from my_library.splitters.cross_validation import CrossValidationSplitter


# pytest my_library/tests/unit/splitters/test_cross_validation.py -v
@pytest.fixture
def sample_data():
    # ダミーの DataFrame と Series を準備
    X = pd.DataFrame({
        'feature1': range(10),
        'feature2': np.linspace(0, 1, 10)
    })
    y = pd.Series(np.arange(10), name='target')
    return X, y


def test_split_return_type_and_length(sample_data):
    X, y = sample_data
    splitter = CrossValidationSplitter(n_splits=5, shuffle=False)
    folds = splitter.split(X, y)
    assert isinstance(folds, list), "split はリストを返すべき"
    assert len(folds) == 5, "n_splits=5 の場合、5つのfoldを返すべき"

    for train, valid in folds:
        X_train, y_train = train
        X_valid, y_valid = valid
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_valid, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_valid, pd.Series)


def test_default_n_splits_and_fold_sizes(sample_data):
    X, y = sample_data
    splitter = CrossValidationSplitter()  # デフォルト n_splits=5
    folds = splitter.split(X, y)
    assert len(folds) == 5, "デフォルトで5foldを返すはず"

    # 10サンプルを5分割すると各foldの検証用は2件、学習用は8件
    for (_, _), (X_valid, y_valid) in folds:
        assert len(X_valid) == 2
        assert len(y_valid) == 2
        assert len(folds[0][0][0]) == 8  # 学習用の長さチェック
        assert len(folds[0][0][1]) == 8


def test_custom_n_splits_fold_sizes(sample_data):
    X, y = sample_data
    splitter = CrossValidationSplitter(n_splits=3, shuffle=False)
    folds = splitter.split(X, y)
    assert len(folds) == 3, "n_splits=3 の場合、3つのfoldを返すべき"

    # 10サンプルを3分割すると、検証用のfoldサイズは順に [4, 3, 3]
    val_sizes = [len(valid[0]) for _, valid in folds]
    assert val_sizes == [4, 3, 3]


def test_reproducible_splits(sample_data):
    X, y = sample_data
    s1 = CrossValidationSplitter(n_splits=5, shuffle=True, random_state=123)
    s2 = CrossValidationSplitter(n_splits=5, shuffle=True, random_state=123)
    folds1 = s1.split(X, y)
    folds2 = s2.split(X, y)

    for ((X_tr1, y_tr1), (X_val1, y_val1)), ((X_tr2, y_tr2), (X_val2, y_val2)) \
        in zip(folds1, folds2, strict=False):
        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_series_equal(y_tr1, y_tr2)
        pd.testing.assert_frame_equal(X_val1, X_val2)
        pd.testing.assert_series_equal(y_val1, y_val2)


def test_groups_ignored(sample_data):
    X, y = sample_data
    groups = pd.Series(['A'] * 10)
    splitter = CrossValidationSplitter(n_splits=5, shuffle=False, random_state=0)

    no_grp = splitter.split(X, y)
    with_grp = splitter.split(X, y, groups=groups)

    # groups を渡しても結果が変わらないこと
    for fold_no, fold_grp in zip(no_grp, with_grp, strict=False):
        (X_tr_no, y_tr_no), (X_val_no, y_val_no) = fold_no
        (X_tr_gr, y_tr_gr), (X_val_gr, y_val_gr) = fold_grp

        pd.testing.assert_frame_equal(X_tr_no, X_tr_gr)
        pd.testing.assert_series_equal(y_tr_no, y_tr_gr)
        pd.testing.assert_frame_equal(X_val_no, X_val_gr)
        pd.testing.assert_series_equal(y_val_no, y_val_gr)


def test_array_inputs():
    X = np.arange(50).reshape(25, 2)
    y = np.arange(25)
    splitter = CrossValidationSplitter(n_splits=5, shuffle=False)
    folds = splitter.split(X, y)
    assert len(folds) == 5

    X_train, y_train = folds[0][0]
    X_valid, y_valid = folds[0][1]

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_valid, np.ndarray)
    assert isinstance(y_valid, np.ndarray)
    assert X_train.shape[0] == 20
    assert X_valid.shape[0] == 5


def test_mismatched_lengths():
    X = pd.DataFrame({'a': range(5)})
    y = pd.Series(range(6))
    splitter = CrossValidationSplitter()
    with pytest.raises(ValueError):
        splitter.split(X, y)

import numpy as np
import pandas as pd
import pytest

from my_library.splitters.holdout import HoldoutSplitter


# pytest my_library/tests/unit/splitters/test_holdout.py -v
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
    splitter = HoldoutSplitter(test_size=0.3, random_state=0)
    folds = splitter.split(X, y)
    assert isinstance(folds, list), "split はリストを返すべき"
    assert len(folds) == 1, "ホールドアウトは1つのfoldのみを返すべき"

    (X_train, y_train), (X_valid, y_valid) = folds[0]
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_valid, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_valid, pd.Series)


def test_default_parameters(sample_data):
    X, y = sample_data
    splitter = HoldoutSplitter()  # デフォルト test_size=0.2, random_state=42
    (X_train, y_train), (X_valid, y_valid) = splitter.split(X, y)[0]
    assert len(X_valid) == 2, "デフォルトの20%で10サンプル中2サンプルが検証用になるはず"
    assert len(X_train) == 8


def test_train_valid_size(sample_data):
    X, y = sample_data
    splitter = HoldoutSplitter(test_size=0.25, random_state=42)
    (X_train, y_train), (X_valid, y_valid) = splitter.split(X, y)[0]
    # 10 * 0.25 = 2.5 → 生データでは ceil(2.5)=3 サンプルが検証用
    assert len(X_valid) == 3
    assert len(X_train) == 7


def test_reproducible_splits(sample_data):
    X, y = sample_data
    s1 = HoldoutSplitter(test_size=0.2, random_state=123)
    s2 = HoldoutSplitter(test_size=0.2, random_state=123)
    (X_tr1, y_tr1), (X_val1, y_val1) = s1.split(X, y)[0]
    (X_tr2, y_tr2), (X_val2, y_val2) = s2.split(X, y)[0]

    pd.testing.assert_frame_equal(X_tr1, X_tr2)
    pd.testing.assert_series_equal(y_tr1, y_tr2)
    pd.testing.assert_frame_equal(X_val1, X_val2)
    pd.testing.assert_series_equal(y_val1, y_val2)


def test_groups_ignored(sample_data):
    X, y = sample_data
    groups = pd.Series(['A'] * 10)
    splitter = HoldoutSplitter(test_size=0.2, random_state=0)

    no_grp = splitter.split(X, y)
    with_grp = splitter.split(X, y, groups=groups)

    # groups を渡しても結果が変わらないこと
    pd.testing.assert_frame_equal(no_grp[0][0][0], with_grp[0][0][0])
    pd.testing.assert_series_equal(no_grp[0][0][1], with_grp[0][0][1])
    pd.testing.assert_frame_equal(no_grp[0][1][0], with_grp[0][1][0])
    pd.testing.assert_series_equal(no_grp[0][1][1], with_grp[0][1][1])


def test_array_inputs():
    X = np.arange(50).reshape(25, 2)
    y = np.arange(25)
    splitter = HoldoutSplitter(test_size=0.2, random_state=1)
    (X_train, y_train), (X_valid, y_valid) = splitter.split(X, y)[0]

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_valid, np.ndarray)
    assert isinstance(y_valid, np.ndarray)
    assert X_train.shape[0] == 20
    assert X_valid.shape[0] == 5


def test_mismatched_lengths():
    X = pd.DataFrame({'a': range(5)})
    y = pd.Series(range(6))
    splitter = HoldoutSplitter()
    with pytest.raises(ValueError):
        splitter.split(X, y)

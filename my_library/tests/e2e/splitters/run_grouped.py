import numpy as np
import pandas as pd

from my_library.splitters.grouped import GroupKFoldSplitter

# ── サンプルデータ作成 ──
# 12件のサンプル、グループ A～F をそれぞれ2つずつ
X = pd.DataFrame({
    'feature1': np.random.randn(12),
    'feature2': np.arange(12)
})
y = pd.Series(np.random.randint(0, 2, size=12), name='target')
groups = pd.Series(list('AABBCCDDEEFF'), name='group')

print("全データ")
print(pd.concat([X, y, groups], axis=1), "\n")

# ── splitter 初期化 & split 実行 ──
splitter = GroupKFoldSplitter(n_splits=3)
folds = splitter.split(X, y, groups)

# ── 各フォールドごとに確認 ──
for fold_idx, ((X_tr, _), (X_val, _)) in enumerate(folds, start=1):
    train_idx = list(X_tr.index)
    valid_idx = list(X_val.index)
    train_groups = set(groups.iloc[train_idx])
    valid_groups = set(groups.iloc[valid_idx])

    print(f"--- Fold {fold_idx} ---")
    print(f"Train indices      : {train_idx}")
    print(f"Valid indices      : {valid_idx}")
    print(f"Train groups       : {sorted(train_groups)}")
    print(f"Valid groups       : {sorted(valid_groups)}")

    # グループの重複がないことをアサート
    assert train_groups.isdisjoint(valid_groups), "Group leakage detected!"
    print("No group leakage ✔\n")

# ── 長さ不一致の例外確認 ──
try:
    bad_y = y[:-1]  # 長さをズラす
    splitter.split(X, bad_y, groups)
except ValueError as e:
    print("Expected error for mismatched lengths:", e)

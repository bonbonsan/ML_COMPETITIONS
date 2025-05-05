import numpy as np
import pandas as pd

from my_library.splitters.stratified_grouped import StratifiedGroupKFoldSplitter

# ── サンプルデータ作成 ──
# 12件のサンプル、グループ A–F をそれぞれ2つずつ
# グループ A,B,C はクラス0、D,E,F はクラス1
groups = pd.Series(list('AABBCCDDEEFF'), name='group')
y = pd.Series([0]*6 + [1]*6, name='target')
X = pd.DataFrame({
    'feature1': np.random.randn(12),
    'feature2': np.arange(12)
})
print("Full dataset:")
print(pd.concat([X, y, groups], axis=1), "\n")

# ── splitter 初期化 & split 実行 ──
splitter = StratifiedGroupKFoldSplitter(n_splits=3, random_state=0)
folds = splitter.split(X, y, groups)

# ── 各フォールドごとに確認 ──
for i, ((X_tr, _), (X_val, y_val)) in enumerate(folds, start=1):
    train_idx = list(X_tr.index)
    valid_idx = list(X_val.index)
    train_groups = set(groups.loc[train_idx])
    valid_groups = set(groups.loc[valid_idx])
    val_counts = y_val.value_counts().to_dict()

    print(f"--- Fold {i} ---")
    print(f"Train indices : {train_idx}")
    print(f"Valid indices : {valid_idx}")
    print(f"Train groups  : {sorted(train_groups)}")
    print(f"Valid groups  : {sorted(valid_groups)}")
    print(f"Valid class distribution: {val_counts}")

    # グループの重複がないことをチェック
    assert train_groups.isdisjoint(valid_groups), "Group leakage detected!"
    # 各フォールドにクラス0,1が同数含まれていることをチェック
    assert val_counts.get(0, 0) == 2 and val_counts.get(1, 0) == 2, "Stratification failed!"
    print("✔ No leakage, stratification OK\n")

# ── 再現性の確認 ──
splitter_a = StratifiedGroupKFoldSplitter(n_splits=3, random_state=123)
splitter_b = StratifiedGroupKFoldSplitter(n_splits=3, random_state=123)
folds_a = splitter_a.split(X, y, groups)
folds_b = splitter_b.split(X, y, groups)
print("Reproducibility check:")
for ((_, _), (Xa_val, _)), ((_, _), (Xb_val, _)) in zip(folds_a, folds_b, strict=False):
    print("  Fold identical indices?", Xa_val.index.tolist() == Xb_val.index.tolist())
print()

# ── 長さ不一致で例外発生 ──
print("Mismatched lengths check:")
try:
    y_bad = y[:-1]  # 長さをずらす
    splitter.split(X, y_bad, groups)
except ValueError as e:
    print("  Expected ValueError:", e)

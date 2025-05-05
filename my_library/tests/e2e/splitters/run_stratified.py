import numpy as np
import pandas as pd

from my_library.splitters.stratified import StratifiedKFoldSplitter

# ── サンプルデータ作成 ──
# 10サンプル、クラス0が6件、クラス1が4件の不均衡データ
X = pd.DataFrame({
    'feat': np.arange(10)
})
y = pd.Series([0]*6 + [1]*4, name='target')
print("Full dataset:")
print(pd.concat([X, y], axis=1), "\n")

# ── splitter 初期化 & split 実行 ──
splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=0)
folds = splitter.split(X, y)

# ── 各フォールドごとに確認 ──
for i, ((X_tr, _), (X_val, y_val)) in enumerate(folds, start=1):
    train_idx = X_tr.index.tolist()
    valid_idx = X_val.index.tolist()
    val_counts = y_val.value_counts().to_dict()

    print(f"--- Fold {i} ---")
    print(f"Train indices: {train_idx}")
    print(f"Valid indices: {valid_idx}")
    print(f"Valid class distribution: {val_counts}")
    # each validation fold should have proportionate class ratios
    total_val = len(y_val)
    ratio0 = val_counts.get(0, 0) / total_val
    ratio1 = val_counts.get(1, 0) / total_val
    print(f"  ratio of class 0: {ratio0:.2f}, class 1: {ratio1:.2f}\n")

# ── 再現性の確認 ──
print("Reproducibility check:")
split_a = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=42)
split_b = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=42)
folds_a = split_a.split(X, y)
folds_b = split_b.split(X, y)
for idx, (((_, _), (Xa_val, _)), ((_, _), (Xb_val, _))) \
    in enumerate(zip(folds_a, folds_b, strict=False), start=1):
    same = Xa_val.index.tolist() == Xb_val.index.tolist()
    print(f" Fold {idx} identical validation indices? {same}")
print()

# ── groups 引数無視の確認 ──
print("Groups ignored check:")
dummy_groups = pd.Series(['G'] * len(X), name='group')
folds_no_grp = splitter.split(X, y)
folds_with_grp = splitter.split(X, y, groups=dummy_groups)
for idx, (((_, _), (X0_val, _)), ((_, _), (X1_val, _))) \
    in enumerate(zip(folds_no_grp, folds_with_grp, strict=False), start=1):
    same = X0_val.index.tolist() == X1_val.index.tolist()
    print(f" Fold {idx} identical when passing groups? {same}")

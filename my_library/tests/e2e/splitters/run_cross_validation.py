import numpy as np
import pandas as pd

from my_library.splitters.cross_validation import CrossValidationSplitter

# ── サンプルデータ作成 ──
X = pd.DataFrame({
    'feature1': np.arange(10),
    'feature2': np.linspace(0, 1, 10)
})
y = pd.Series(np.arange(10), name='target')
print("Full dataset:")
print(pd.concat([X, y], axis=1), "\n")

# ── 1) デフォルト設定での分割 ──
# n_splits=5, shuffle=True, random_state=42
splitter_default = CrossValidationSplitter()
folds_default = splitter_default.split(X, y)
print("Default CV split (n_splits=5, shuffle=True):")
for i, ((X_tr, _), (X_val, _)) in enumerate(folds_default):
    print(f"  Fold {i}:")
    print(f"    Train indices: {list(X_tr.index)}")
    print(f"    Valid indices: {list(X_val.index)}")
    print(f"    Train size: {len(X_tr)}, Valid size: {len(X_val)}")
print()

# ── 2) カスタムパラメータでの分割 ──
# n_splits=3, shuffle=False
splitter_custom = CrossValidationSplitter(n_splits=3, shuffle=False)
folds_custom = splitter_custom.split(X, y)
print("Custom CV split (n_splits=3, shuffle=False):")
for i, ((X_tr, _), (X_val, _)) in enumerate(folds_custom):
    print(f"  Fold {i}:")
    print(f"    Train indices: {list(X_tr.index)}")
    print(f"    Valid indices: {list(X_val.index)}")
    print(f"    Train size: {len(X_tr)}, Valid size: {len(X_val)}")
print()

# ── 3) 再現性の確認 ──
splitter_a = CrossValidationSplitter(n_splits=5, shuffle=True, random_state=123)
splitter_b = CrossValidationSplitter(n_splits=5, shuffle=True, random_state=123)
folds_a = splitter_a.split(X, y)
folds_b = splitter_b.split(X, y)
same = all(
    list(folds_a[i][1][1].index) == list(folds_b[i][1][1].index)
    for i in range(len(folds_a))
)
print("Reproducibility check:")
print("  Splits identical across all folds?", same, "\n")

# ── 4) groups 引数を渡しても結果は変わらない ──
groups = pd.Series(['G'] * len(X), name='group')
folds_no_grp = splitter_default.split(X, y)
folds_with_grp = splitter_default.split(X, y, groups=groups)
identical = all(
    list(folds_no_grp[i][1][0].index) == list(folds_with_grp[i][1][0].index) \
    and list(folds_no_grp[i][1][1].index) == list(folds_with_grp[i][1][1].index) \
        for i in range(len(folds_no_grp))
    )
print("Groups ignored check:")
print("  Splits identical when passing groups?", identical, "\n")

# ── 5) NumPy 配列の入力もサポート ──
X_arr = np.arange(30).reshape(15, 2)
y_arr = np.arange(15)
folds_arr = splitter_default.split(X_arr, y_arr)
print("NumPy array input:")
for i, ((X_tr_arr, _), (X_val_arr, _)) in enumerate(folds_arr):
    print(f"  Fold {i}: Train shape: {X_tr_arr.shape}, Valid shape: {X_val_arr.shape}")
print()

# ── 6) 長さ不一致時の例外 ──
print("Mismatched lengths check:")
try:
    y_bad = y[:-1]
    splitter_default.split(X, y_bad)
except ValueError as e:
    print("  Caught expected ValueError:", e)

import numpy as np
import pandas as pd

from my_library.splitters.holdout import HoldoutSplitter

# ── サンプルデータ作成 ──
# 10件のサンプルを持つDataFrameとSeriesを生成
X = pd.DataFrame({
    'feature1': np.arange(10),
    'feature2': np.linspace(0, 1, 10)
})
y = pd.Series(np.arange(10), name='target')
print("Full dataset:")
print(pd.concat([X, y], axis=1), "\n")

# ── 1) デフォルト設定での分割 ──
splitter_default = HoldoutSplitter()  # test_size=0.2, random_state=42
((X_tr_def, y_tr_def), (X_val_def, y_val_def)) = splitter_default.split(X, y)[0]
print("Default split (test_size=0.2):")
print(f"  Train indices: {list(X_tr_def.index)}")
print(f"  Valid indices: {list(X_val_def.index)}")
print(f"  Train size: {len(X_tr_def)}, Valid size: {len(X_val_def)}\n")

# ── 2) カスタムパラメータでの分割 ──
splitter_custom = HoldoutSplitter(test_size=0.3, random_state=0)
((X_tr_c, y_tr_c), (X_val_c, y_val_c)) = splitter_custom.split(X, y)[0]
print("Custom split (test_size=0.3, random_state=0):")
print(f"  Train indices: {list(X_tr_c.index)}")
print(f"  Valid indices: {list(X_val_c.index)}")
print(f"  Train size: {len(X_tr_c)}, Valid size: {len(X_val_c)}\n")

# ── 3) 再現性の確認 ──
splitter_a = HoldoutSplitter(test_size=0.25, random_state=123)
splitter_b = HoldoutSplitter(test_size=0.25, random_state=123)
_, (Xa_val, _) = splitter_a.split(X, y)[0]
_, (Xb_val, _) = splitter_b.split(X, y)[0]
print("Reproducibility check:")
print("  Splits identical?" , Xa_val.index.tolist() == Xb_val.index.tolist(), "\n")

# ── 4) groups 引数を渡しても結果は変わらない ──
groups = pd.Series(['G'] * len(X), name='group')
split_no_grp = splitter_default.split(X, y)
split_with_grp = splitter_default.split(X, y, groups=groups)
print("Groups ignored check:")
print("  Splits identical when passing groups?",
      split_no_grp[0][1][0].index.tolist() == split_with_grp[0][1][0].index.tolist(),
      "\n")

# ── 5) NumPy 配列の入力もサポート ──
X_arr = np.arange(30).reshape(15, 2)
y_arr = np.arange(15)
split_arr = splitter_default.split(X_arr, y_arr)[0]
print("NumPy array input:")
print(f"  Train shape: {split_arr[0][0].shape}, Valid shape: {split_arr[1][0].shape}\n")

# ── 6) 長さ不一致時の例外 ──
print("Mismatched lengths check:")
try:
    y_bad = y[:-1]
    splitter_default.split(X, y_bad)
except ValueError as e:
    print("  Caught expected ValueError:", e)

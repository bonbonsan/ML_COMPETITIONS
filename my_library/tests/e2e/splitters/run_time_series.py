import numpy as np
import pandas as pd

from my_library.splitters.time_series import TimeSeriesSplitter

# ── サンプルデータ作成 ──
dates = pd.date_range(start="2021-01-01", periods=12, freq="D")
X = pd.DataFrame({
    "date": dates,
    "feature": np.arange(12)
})
y = pd.Series(np.arange(12), name="target")

print("Full dataset:")
print(X.join(y), "\n")

# ── Expanding ウィンドウ ──
print("===== Expanding method =====")
splitter_exp = TimeSeriesSplitter(n_splits=3, method="expanding")
folds_exp = splitter_exp.split(X, y)
for i, ((X_tr, _), (X_val, _)) in enumerate(folds_exp, start=1):
    print(f"Fold {i}:")
    print("  Train dates:", X_tr["date"].min().date(), "→", X_tr["date"].max().date())
    print("  Valid dates:", X_val["date"].min().date(), "→", X_val["date"].max().date())
    print("  Train size:", len(X_tr), "Valid size:", len(X_val), "\n")

# ── Rolling ウィンドウ ──
print("===== Rolling method =====")
# max_train_size=5, test_size=2 で固定長トレーニング＋テスト窓
splitter_roll = TimeSeriesSplitter(
    n_splits=3,
    method="rolling",
    max_train_size=4,
    test_size=2
)
folds_roll = splitter_roll.split(X, y)
for i, ((X_tr, _), (X_val,_)) in enumerate(folds_roll, start=1):
    print(f"Fold {i}:")
    print("  Train dates:", X_tr["date"].min().date(), "→", X_tr["date"].max().date())
    print("  Valid dates:", X_val["date"].min().date(), "→", X_val["date"].max().date())
    print("  Train size:", len(X_tr), "Valid size:", len(X_val), "\n")

# ── 長さ不一致チェック ──
print("===== Mismatched lengths check =====")
try:
    y_bad = y[:-1]
    TimeSeriesSplitter(n_splits=2, method="expanding").split(X, y_bad)
except ValueError as e:
    print("Caught expected ValueError:", e)

import polars as pl
from sklearn.datasets import load_iris

# アヤメデータセットを読み込み、pandasからpolarsへ変換
iris = load_iris(as_frame=True)
df_pd = iris.frame
df_pd["target"] = iris.target
df = pl.from_pandas(df_pd)

# --- データの確認 ---
print("-- データの確認 ---")
print(df.head())

# --- 基本統計量 ---
print("-- 基本統計量 ---")
print(df.describe())

# --- サブセット（条件抽出） ---
print("-- サブセット（条件抽出） ---")
filtered = df.filter(pl.col("sepal length (cm)") > 5.0)
print(filtered)

# --- ランダムサンプリング ---
print("-- ランダムサンプリング ---")
sampled = df.sample(n=5)
print(sampled)

# --- 特定の列を選択 ---
print("-- 特定の列を選択 ---")
selected_cols = df.select([
    "sepal length (cm)",
    "species" if "species" in df.columns else "target"
])
print(selected_cols)

# --- 正規表現で列選択（'sepal'で始まる列） ---
print("-- 正規表現で列選択（'sepal'で始まる列） ---")
regex_select = df.select(pl.col("^sepal.*$"))
print(regex_select)

# --- 行・列のスライス ---
print("-- 行・列のスライス ---")
slice_rows = df[2:5]
slice_cols = df[:, [0, 2]]
print(slice_rows)
print(slice_cols)

# --- ソート ---
print("-- ソート ---")
sorted_df = df.sort(by="sepal width (cm)", descending=True)
print(sorted_df.head())

# --- 列名の変更 ---
print("-- 列名の変更 ---")
renamed_df = df.rename({"sepal length (cm)": "sepal_len"})
print(renamed_df.columns)

# --- 列の追加（新しい列: petal面積） ---
print("-- 列の追加（新しい列: petal面積） ---")
df = df.with_columns(
    (pl.col("petal length (cm)") * pl.col("petal width (cm)"))
    .alias("petal_area")
)
print(df.select(["petal_area"]).head())

# --- グループ化と集約（targetごと） ---
print("-- グループ化と集約（targetごと） ---")
grouped = df.group_by("target").agg([
    pl.mean("sepal length (cm)").alias("avg_sepal_len"),
    pl.max("petal_area").alias("max_petal_area"),
])
print(grouped)

# --- 欠損値処理（存在しないが例示として） ---
print("-- 欠損値処理（存在しないが例示として） ---")
df_with_nan = df.with_columns(pl.lit(None).alias("missing_col"))
print(df_with_nan.fill_null(0).select(["missing_col"]))

# --- ローリング処理（例として petal_area） ---
print("-- ローリング処理（例として petal_area） ---")
rolling = df.select([
    pl.col("petal_area")
    .rolling_mean(window_size=3)
    .alias("rolling_avg")
])
print(rolling.head())

# --- ウィンドウ関数 ---
print("-- ウィンドウ関数 ---")
window_df = df.select([
    "target",
    pl.col("petal_area")
    .sum()
    .over("target")
    .alias("sum_petal_area_by_target")
])
print(window_df)

# --- 行数・列数・ユニーク数 ---
print("-- 行数・列数・ユニーク数 ---")
print("shape:", df.shape)
print("unique targets:", df["target"].n_unique())

# --- サンプルデータの作成 ---
print("-- サンプルデータの作成 ---")
df1 = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

df2 = pl.DataFrame({
    "id": [4, 5],
    "name": ["David", "Eva"]
})

df3 = pl.DataFrame({
    "age": [25, 30, 35],
    "city": ["Tokyo", "Osaka", "Nagoya"]
})

df4 = pl.DataFrame({
    "id": [2, 3, 6],
    "score": [85, 90, 70]
})

# --- 縦方向の結合（行追加） ---
print("-- 縦方向の結合（行追加） ---")
concat_rows = pl.concat([df1, df2])
print("▼ concat rows ▼")
print(concat_rows)

# --- 横方向の結合（列追加） ---
print("-- 横方向の結合（列追加） ---")
concat_cols = pl.concat([df1, df3], how="horizontal")
print("▼ concat columns ▼")
print(concat_cols)

# --- 内部結合（共通の'id'がある行だけ） ---
print("-- 内部結合（共通の'id'がある行だけ） ---")
inner_join = df1.join(df4, on="id", how="inner")
print("▼ inner join ▼")
print(inner_join)

# --- 左外部結合（df1の行をすべて保持） ---
print("-- 左外部結合（df1の行をすべて保持） ---")
left_join = df1.join(df4, on="id", how="left")
print("▼ left join ▼")
print(left_join)

# --- 外部結合（両方の全行を保持） ---
print("-- 外部結合（両方の全行を保持） ---")
outer_join = df1.join(df4, on="id", how="outer")
print("▼ outer join ▼")
print(outer_join)

# --- アンチ結合（df1にあってdf4にない行） ---
print("-- アンチ結合（df1にあってdf4にない行） ---")
anti_join = df1.join(df4, on="id", how="anti")
print("▼ anti join ▼")
print(anti_join)

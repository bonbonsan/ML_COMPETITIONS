import seaborn as sns

import my_library.utils.viz as viz_utils


@viz_utils.savefig_decorator
def plot_count_pairs(data_df, feature, title, hue="set"):
    """
    Plot a count plot of a categorical feature grouped by a hue column.

    Parameters:
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing the data to plot.
    feature : str
        The name of the categorical feature column to be plotted on the x-axis.
    title : str
        The title suffix for the plot, typically indicating the feature name.
    hue : str, optional (default='set')
        The column used to group and color the bars (e.g., train/test or class labels).
    save_path : str, optional
        File path to save the figure as PNG. If not provided, the plot is only displayed.

    Returns:
    -------
    matplotlib.figure.Figure
        The created matplotlib figure.
    """
    fig, ax = viz_utils.create_figure()
    sns.countplot(x=feature, data=data_df, hue=hue, palette=viz_utils.COLOR_LIST, ax=ax)
    ax.set_title(f"Number of passengers / {title}")
    viz_utils.apply_grid_style(ax)
    return fig


@viz_utils.savefig_decorator
def plot_distribution_pairs(data_df, feature, title, hue="set"):
    """
    Plot overlaid histograms of a numerical feature split by a hue column.

    Parameters:
    ----------
    data_df : pd.DataFrame
        The input DataFrame containing the data to plot.
    feature : str
        The name of the numerical feature whose distribution is to be plotted.
    title : str
        The title suffix for the plot, typically indicating the feature name.
    hue : str, optional (default='set')
        The column used to split and color the distributions.
    save_path : str, optional
        File path to save the figure as PNG. If not provided, the plot is only displayed.

    Returns:
    -------
    matplotlib.figure.Figure
        The created matplotlib figure.
    """
    fig, ax = viz_utils.create_figure()
    unique_values = data_df[hue].unique()

    for i, h in enumerate(unique_values):
        sns.histplot(
            data_df.loc[data_df[hue] == h, feature],
            color=viz_utils.COLOR_LIST[i % len(viz_utils.COLOR_LIST)],
            label=h,
            ax=ax,
            kde=False
        )
    ax.set_title(f"Number of passengers / {title}")
    ax.legend()
    viz_utils.apply_grid_style(ax)
    return fig


if __name__ == "__main__":
    import os

    import numpy as np
    import pandas as pd

    # --- 出力ディレクトリの作成 ---
    output_dir = "my_library/output"
    os.makedirs(output_dir, exist_ok=True)

    # --- ダミーデータの生成 ---
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        "Sex": np.random.choice(["male", "female"], size=n_samples),
        "Age": np.random.normal(loc=30, scale=10, size=n_samples).clip(0, 80),
        "set": np.random.choice(["train", "test"], size=n_samples, p=[0.7, 0.3])
    })

    # --- グラフの出力 ---
    plot_count_pairs(
        data_df=df,
        feature="Sex",
        title="Sex Distribution",
        hue="set",
        save_path=os.path.join(output_dir, "sex_count.png")
    )

    plot_distribution_pairs(
        data_df=df,
        feature="Age",
        title="Age Distribution",
        hue="set",
        save_path=os.path.join(output_dir, "age_distribution.png")
    )

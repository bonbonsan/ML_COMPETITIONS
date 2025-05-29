from functools import wraps

import matplotlib.pyplot as plt
import seaborn as sns

# === 定数 ===

COLOR_LIST = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
"""
COLOR_LIST : List[str]
    A predefined list of hex colors used consistently for plotting.
"""

# === Decorator ===

def savefig_decorator(func):
    """
    Decorator to save a matplotlib figure as a PNG file if 'save_path' is provided.
    Always shows the plot via plt.show() at the end.

    Parameters:
    ----------
    func : Callable
        A function that returns a matplotlib Figure object.

    Returns:
    -------
    Callable
        Wrapped function that optionally saves the figure and always shows it.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        save_path = kwargs.pop("save_path", None)
        fig = func(*args, **kwargs)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return fig
    return wrapper

# === Helper Functions ===

def create_figure(figsize=(8, 4)):
    """
    Create a matplotlib figure and axis with a default size.

    Parameters:
    ----------
    figsize : tuple of (float, float), default=(8, 4)
        Size of the figure in inches.

    Returns:
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axis of the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return fig, ax

def apply_grid_style(ax):
    """
    Apply consistent grid styling to a matplotlib axis.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The axis to which the grid style will be applied.

    Returns:
    -------
    None
    """
    ax.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")

def init_plotting_style():
    """
    Set global plotting style using seaborn and matplotlib parameters.

    This function should be called once at the beginning of a notebook or script.

    Returns:
    -------
    None
    """
    sns.set_theme(style="whitegrid", palette=COLOR_LIST)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 100

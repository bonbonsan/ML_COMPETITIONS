from typing import Optional

import pandas as pd
import yfinance as yf


def load_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    columns: Optional[list[str]] = None,
    target: Optional[str] = "Close",
    dropna: bool = True
) -> pd.DataFrame:
    """Download stock price data using yfinance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        period (str): Lookback period (e.g., '1y', '6mo', '5d').
        interval (str): Data interval (e.g., '1d', '1wk', '1mo').
        columns (list[str], optional): Columns to include. Default: all.
        target (str, optional): Column to rename as 'target'.
        dropna (bool): Whether to drop rows with NaN values.

    Returns:
        pd.DataFrame: DataFrame with date index and specified columns.
    """
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if columns:
        df = df[columns]

    if target and target in df.columns:
        df = df.rename(columns={target: "target"})

    if dropna:
        df = df.dropna()

    return df


if __name__ == "__main__":
    # TODO: ValueError: No data found for ticker: xxxx
    df = load_stock_data(
        "MSFT",
        period="6mo",
        interval="1d",
        columns=["Open", "High", "Low", "Close"],
        target="Close"
    )
    print(df.head())

import pandas as pd


def create_sequences(df: pd.DataFrame, col: str, time_steps: int = 5
                     ) -> tuple[pd.DataFrame, pd.Series]:
    """Convert univariate time-series into supervised learning format with lag features.

    This function creates a dataset where each row consists of `time_steps` consecutive
    values of the specified column, and the target is the next value after the sequence.

    Args:
        df (pd.DataFrame): Original time-series DataFrame with a datetime index.
        col (str): Column name to convert into sequences.
        time_steps (int): Number of time steps to use for each sequence.

    Returns:
        tuple[pd.DataFrame, pd.Series]:
            - Feature DataFrame of shape (N - time_steps, time_steps)
            - Target Series of length (N - time_steps)
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    values = df[col].values
    if len(values) <= time_steps:
        raise ValueError("Not enough data to create sequences.")

    X, y = [], []
    for i in range(len(values) - time_steps):
        X.append(values[i:i + time_steps])
        y.append(values[i + time_steps])

    X_df = pd.DataFrame(X, index=df.index[time_steps:])
    y_series = pd.Series(y, index=df.index[time_steps:], name=col)

    return X_df, y_series


if __name__ == "__main__":
    from my_library.utils.data_loader import load_sample_data

    df_airline = load_sample_data(name="airline", task="regression")
    df_airline.set_index("Month", inplace=True)

    # シーケンス化
    X_seq, y_seq = create_sequences(df_airline, col="Passengers", time_steps=5)

    print("---df_airline---")
    print(df_airline)
    print("---X_seq---")
    print(X_seq)
    print("---y_seq---")
    print(y_seq)

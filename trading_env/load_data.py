import pandas as pd

def load_returns(path):
    """
    Loading the data csv and pivots it into:
    R: numpy matrix of shape [T, A]
    dates: ordered list of dates
    assets: ordered list of assets
    """

    # Reading the CSV file into a DataFrame
    df = pd.read_csv(path)

    # Converting date column (string tp datetime object using .to_datatime)
    df["date"] = pd.to_datetime(df["date"])

    # Sorting by date then asset, just to be safe
    df = df.sort_values(["date", "asset"])

    # Pivot from long format to matrix format:
    # rows = dates, columns = assets, values = returns
    pivot = df.pivot(index="date", columns="asset", values="ret")

    # Extract dates and assets lists
    dates = list(pivot.index)        # list of all dates
    assets = list(pivot.columns)     # list of all tickers (SPY, QQQ...)

    # Convert pivot table to a numpy matrix
    R = pivot.values                 # shape [T, A]

    return R, dates, assets

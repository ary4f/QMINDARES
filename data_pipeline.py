#importing all the shite we need
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIG =====
TICKERS = ["SPY", "TLT", "GLD", "QQQ"]
START = "2016-01-01"
END   = "2025-10-31"
OUT_PATH = "data/prices_returns.csv"
PLOT = False  # set True locally if you want to see the price chart

# ===== Loader =====
def load_prices(tickers, start, end):
    """
    Download adjusted close prices from Yahoo Finance and return
    a LONG DataFrame with columns: date, asset, close.
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,  # we'll use Adj Close explicitly
        progress=False,
    )

    # If multi-index columns, pick "Adj Close"
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"]
    else:
        prices = raw  # fallback, but yfinance should give MultiIndex

    # Index → column
    prices = prices.reset_index().rename(columns={"Date": "date"})

    # Long format
    long_df = prices.melt(
        id_vars="date",
        var_name="asset",
        value_name="close",
    )

    # Clean up
    long_df = long_df.dropna(subset=["close"])
    long_df["date"] = pd.to_datetime(long_df["date"]).dt.normalize()

    return long_df

# ===== Calendar Aligner =====
def align_calendar(df, tickers):
    """
    Keep only dates for which ALL assets have data.
    Returns df sorted by date, asset.
    """
    counts = df.groupby("date")["asset"].nunique()
    valid_dates = counts[counts == len(tickers)].index

    aligned = (
        df[df["date"].isin(valid_dates)]
        .sort_values(["date", "asset"])
        .reset_index(drop=True)
    )

    # fail fast: every remaining date must have all tickers
    assert aligned.groupby("date")["asset"].nunique().eq(len(tickers)).all(), \
        "Calendar not fully aligned across assets."

    return aligned

# ===== Returns =====
def to_returns(df):
    """
    Given long df[date, asset, close], compute log returns per asset.
    Output columns: date, asset, close, ret
    """
    df = df.sort_values(["asset", "date"]).copy()

    def _asset_returns(g):
        g = g.sort_values("date").copy()
        g["ret"] = np.log(g["close"] / g["close"].shift(1))
        return g.iloc[1:]  # drop first row (NaN ret)

    out = df.groupby("asset", group_keys=False).apply(_asset_returns).reset_index(drop=True)

    # safety checks
    if out["ret"].isna().any():
        raise ValueError("NaNs found in returns after computation.")
    if np.isinf(out["ret"]).any():
        raise ValueError("Inf values found in returns after computation.")

    # stable dtypes and column order for the env
    out["close"] = out["close"].astype(np.float32)
    out["ret"]   = out["ret"].astype(np.float32)
    out = out[["date", "asset", "close", "ret"]]

    # prove identical per-asset date counts
    per_asset_days = out.groupby("asset")["date"].nunique()
    assert per_asset_days.nunique() == 1, f"Per-asset date counts differ:\n{per_asset_days}"

    return out

# ===== Summarize + writer =====
def summarize_and_save(df, out_path):
    """
    Save df to CSV and print a summary:
    - assets
    - date range
    - rows
    - per-asset mean & vol of returns
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)

    assets = sorted(df["asset"].unique())
    date_min = df["date"].min()
    date_max = df["date"].max()
    n_rows = len(df)

    print("=== DATA SUMMARY ===")
    print(f"Assets: {assets}")
    print(f"Date range: {pd.to_datetime(date_min).date()} → {pd.to_datetime(date_max).date()}")
    print(f"Total rows: {n_rows}\n")

    print("Per-asset return stats (daily):")
    stats = df.groupby("asset")["ret"].agg(["mean", "std"]).rename(columns={"std": "vol"})
    print(stats, "\n")

    print(f"Saved to: {out_path}")

# ===== Checks & Plots (optional) =====
def plot_prices(df):
    pivot = df.pivot(index="date", columns="asset", values="close")
    plt.figure(figsize=(10, 6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=col)
    plt.title("Adjusted Close Prices")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.tight_layout()
    plt.show()

# ===== Main =====
def main():
    print("Loading prices...")
    prices = load_prices(TICKERS, START, END)
    print(f"Loaded {len(prices)} rows of raw prices.")

    print("Aligning calendar...")
    aligned = align_calendar(prices, TICKERS)
    print(f"After alignment: {len(aligned)} rows.")

    print("Computing log returns...")
    prices_returns = to_returns(aligned)
    print(f"With returns: {len(prices_returns)} rows.")

    print("Summarizing and saving...")
    summarize_and_save(prices_returns, OUT_PATH)

    if PLOT:
        plot_prices(aligned)

if __name__ == "__main__":
    main()

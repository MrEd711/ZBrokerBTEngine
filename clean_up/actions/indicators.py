"""
indicators.py

Indicator calculators that ONLY update AppState (no returns).
- You pass `state: AppState` into each function
- Functions read params directly from `state`
- Functions compute indicator DataFrames from `state.csv_data`
- Each indicator DataFrame includes a 'Date' column copied from state.csv_data['Date']
  aligned by row index (NOT derived from the index).
- Indicators are NOT added to state.csv_data

Updates:
- state.macd_values : pd.DataFrame
- state.rsi_values  : pd.DataFrame
- state.roc_values  : pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from state import AppState


# -----------------------------
# Helpers
# -----------------------------

def _require_csv_data(state: AppState) -> pd.DataFrame:
    df = getattr(state, "csv_data", None)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("state.csv_data must be a pandas DataFrame and not None.")
    if "Date" not in df.columns:
        raise KeyError("state.csv_data must contain a 'Date' column.")
    if "Close" not in df.columns:
        raise KeyError("state.csv_data must contain a 'Close' column.")
    return df


def _numeric_close(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df["Close"], errors="coerce")
    return close


def _attach_date_column(indicator_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Date column from base_df['Date'] to indicator_df, aligned by the indicator_df index.
    This ensures each indicator row gets the corresponding Date from the original csv_data.
    """
    out = indicator_df.copy()
    # Insert Date as first column, aligned by row index
    out.insert(0, "Date", base_df["Date"].iloc[out.index].values)
    return out


def _coerce_int(value: object, name: str) -> int:
    try:
        v = int(value)
    except Exception as e:
        raise ValueError(f"{name} must be an int, got {value!r}.") from e
    if v <= 0:
        raise ValueError(f"{name} must be a positive int, got {v}.")
    return v


# -----------------------------
# Public API (updates state only)
# -----------------------------

def update_macd(state: AppState, dropna: bool = True) -> None:
    """
    MACD:
      MACD line   = EMA(close, fast) - EMA(close, slow)
      Signal line = EMA(MACD line, signal)
      Histogram   = MACD line - Signal line

    Reads from state:
      - state.ena_close_fast   (default 12)
      - state.ena_close_slow   (default 26)
      - state.ena_close_signal (default 9)

    Writes to state:
      - state.macd_values (DataFrame with Date + macd/signal/hist columns)
    """
    df = _require_csv_data(state)
    close = _numeric_close(df)

    fast = _coerce_int(getattr(state, "ena_close_fast", 12), "state.ena_close_fast")
    slow = _coerce_int(getattr(state, "ena_close_slow", 26), "state.ena_close_slow")
    sig  = _coerce_int(getattr(state, "ena_close_signal", 9), "state.ena_close_signal")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - signal_line

    out = pd.DataFrame(
        {
            f"macd_{fast}_{slow}_{sig}": macd_line,
            f"macd_signal_{fast}_{slow}_{sig}": signal_line,
            f"macd_hist_{fast}_{slow}_{sig}": hist,
        }
    )

    if dropna:
        out = out.dropna()

    out = _attach_date_column(out, df)
    state.macd_values = out


def update_rsi(state: AppState, dropna: bool = True) -> None:
    """
    RSI (Wilder / RMA smoothing - standard).

    Reads from state:
      - state.rsi_period (default 14)

    Writes to state:
      - state.rsi_values (DataFrame with Date + rsi_<period>)
    """
    df = _require_csv_data(state)
    close = _numeric_close(df)

    period = _coerce_int(getattr(state, "rsi_period", 14), "state.rsi_period")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    alpha = 1.0 / period  # Wilder smoothing
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    out = pd.DataFrame({f"rsi_{period}": rsi})

    if dropna:
        out = out.dropna()

    out = _attach_date_column(out, df)
    state.rsi_values = out


def update_roc(state: AppState, dropna: bool = True) -> None:
    """
    ROC (%):
      ROC = ((close - close.shift(period)) / close.shift(period)) * 100

    Reads from state:
      - state.roc_period (default 12)

    Writes to state:
      - state.roc_values (DataFrame with Date + roc_<period>)
    """
    df = _require_csv_data(state)
    close = _numeric_close(df)

    period = _coerce_int(getattr(state, "roc_period", 12), "state.roc_period")

    prev = close.shift(period)
    roc_pct = ((close - prev) / prev) * 100.0

    out = pd.DataFrame({f"roc_{period}": roc_pct})

    if dropna:
        out = out.dropna()

    out = _attach_date_column(out, df)
    state.roc_values = out


def update_all_indicators(state: AppState, dropna: bool = True) -> None:
    """
    Convenience function: updates MACD, RSI, ROC.
    If dropna=True, aligns all 3 DataFrames to the same row index intersection,
    so they all contain the same set of Dates/rows.
    """
    update_macd(state, dropna=False)
    update_rsi(state, dropna=False)
    update_roc(state, dropna=False)

    if dropna:
        macd_df = state.macd_values.dropna()
        rsi_df = state.rsi_values.dropna()
        roc_df = state.roc_values.dropna()

        idx = macd_df.index.intersection(rsi_df.index).intersection(roc_df.index)

        state.macd_values = macd_df.loc[idx]
        state.rsi_values = rsi_df.loc[idx]
        state.roc_values = roc_df.loc[idx]

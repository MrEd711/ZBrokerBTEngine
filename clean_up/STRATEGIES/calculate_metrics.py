# metrics.py
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# -----------------------------
# Expected data on state
# -----------------------------
# state.trade_log_df  (or whatever you call it): must have the columns you listed:
#   Date, Cumulative Percentage Returns, Account Balance, Short/Long,
#   Entry Price, Exit Price, PnL, Exit Reason
#
# state.csv_data: full OHLCV asset data (Date/Datetime + Open/High/Low/Close [+ Volume])
#
# This file assumes:
#   - state.csv_data has a datetime column called "Date" (or "Datetime"/"Timestamp") and OHLC columns.
#   - trade log has an entry + exit timestamp if you want TRUE MAE (Maximum Adverse Excursion).
#     Recommended: add "Entry Date" and "Exit Date" columns to the trade log.
#
# If you don’t have entry/exit timestamps, MAE cannot be computed correctly from OHLC alone.

TRADE_REQUIRED_COLS = [
    "Date",
    "Cumulative Percentage Returns",
    "Account Balance",
    "Short/Long",
    "Entry Price",
    "Exit Price",
    "PnL",
    "Exit Reason",
]

# Optional but strongly recommended for MAE/MFE from OHLC:
TRADE_TIME_COLS_CANDIDATES = [
    ("Entry Date", "Exit Date"),
    ("Entry Time", "Exit Time"),
    ("Entry Datetime", "Exit Datetime"),
    ("Entry Timestamp", "Exit Timestamp"),
]

OHLC_DATE_CANDIDATES = ["Date", "Datetime", "Timestamp", "time", "date"]
OHLC_COLS_CANDIDATES = [
    ("Open", "High", "Low", "Close"),
    ("open", "high", "low", "close"),
    ("OPEN", "HIGH", "LOW", "CLOSE"),
]


# -----------------------------
# Helpers
# -----------------------------
def _ensure_trade_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("trade log must be a pandas DataFrame")

    missing = [c for c in TRADE_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Trade log missing required columns: {missing}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if out["Date"].isna().all():
        raise ValueError("Trade log 'Date' could not be parsed into datetimes.")
    out = out.sort_values("Date").reset_index(drop=True)

    for c in ["Cumulative Percentage Returns", "Account Balance", "Entry Price", "Exit Price", "PnL"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["Short/Long"] = out["Short/Long"].astype(str).str.strip().str.lower()
    return out


def _ensure_ohlc_df(ohlc: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(ohlc, pd.DataFrame):
        raise TypeError("state.csv_data must be a pandas DataFrame")

    out = ohlc.copy()

    # Find datetime column
    dt_col = None
    for c in OHLC_DATE_CANDIDATES:
        if c in out.columns:
            dt_col = c
            break
    if dt_col is None:
        raise ValueError(f"state.csv_data needs a datetime column (tried {OHLC_DATE_CANDIDATES}).")

    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    if out[dt_col].isna().all():
        raise ValueError(f"state.csv_data '{dt_col}' could not be parsed into datetimes.")

    out = out.sort_values(dt_col).reset_index(drop=True)

    # Find OHLC columns
    ohlc_cols = None
    for cand in OHLC_COLS_CANDIDATES:
        if all(col in out.columns for col in cand):
            ohlc_cols = cand
            break
    if ohlc_cols is None:
        raise ValueError(f"state.csv_data needs OHLC columns (tried {OHLC_COLS_CANDIDATES}).")

    o, h, l, c = ohlc_cols
    for col in [o, h, l, c]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Standardise to canonical names for downstream logic
    out = out.rename(columns={dt_col: "Date", o: "Open", h: "High", l: "Low", c: "Close"})
    return out[["Date", "Open", "High", "Low", "Close"] + [x for x in out.columns if x not in {"Date","Open","High","Low","Close"}]]


def _infer_periods_per_year(dates: pd.Series) -> int:
    d = pd.to_datetime(dates.dropna())
    if len(d) < 3:
        return 252

    diffs = d.diff().dropna()
    if diffs.empty:
        return 252

    median_days = diffs.median() / np.timedelta64(1, "D")

    if median_days <= 2:
        return 252
    if median_days <= 10:
        return 52
    if median_days <= 40:
        return 12
    return 252


def _returns_from_cum_pct(trade_df: pd.DataFrame) -> pd.Series:
    cum = trade_df["Cumulative Percentage Returns"] / 100.0
    equity = 1.0 + cum
    r = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return r


def _get_entry_exit_time_cols(trade_df: pd.DataFrame) -> Optional[tuple[str, str]]:
    for a, b in TRADE_TIME_COLS_CANDIDATES:
        if a in trade_df.columns and b in trade_df.columns:
            return a, b
    return None


# -----------------------------
# Core metrics (assign to state.*)
# -----------------------------
def sharpe_ratio(state, risk_free_rate_annual: float = 0.0) -> Optional[float]:
    df = _ensure_trade_df(state.backtest_results)
    r = _returns_from_cum_pct(df)
    ppy = _infer_periods_per_year(df["Date"])

    rf_per_period = (1.0 + risk_free_rate_annual) ** (1.0 / ppy) - 1.0
    excess = r - rf_per_period
    std = float(excess.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return None
    return float(excess.mean() / std * np.sqrt(ppy))


def sortino_ratio(state, mar_annual: float = 0.0) -> Optional[float]:
    df = _ensure_trade_df(state.backtest_results)
    r = _returns_from_cum_pct(df)
    ppy = _infer_periods_per_year(df["Date"])

    mar_per_period = (1.0 + mar_annual) ** (1.0 / ppy) - 1.0
    downside = (r - mar_per_period)
    downside = downside[downside < 0]

    dd = float(np.sqrt((downside ** 2).mean())) if len(downside) else 0.0
    if dd == 0.0 or np.isnan(dd):
        return None

    excess_mean = float((r - mar_per_period).mean())
    return float(excess_mean / dd * np.sqrt(ppy))


def max_drawdown(state, use_balance: bool = True) -> Optional[float]:
    df = _ensure_trade_df(state.backtest_results)

    if use_balance:
        equity = df["Account Balance"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if len(equity) == 0:
            return None
    else:
        cum = (df["Cumulative Percentage Returns"] / 100.0).astype(float)
        equity = (1.0 + cum).replace([np.inf, -np.inf], np.nan).dropna().values
        if len(equity) == 0:
            return None

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    mdd = float(np.nanmax(dd)) if len(dd) else np.nan
    return None if np.isnan(mdd) else mdd


def cagr(state, use_balance: bool = True) -> Optional[float]:
    df = _ensure_trade_df(state.backtest_results)

    start_date = df["Date"].dropna().iloc[0]
    end_date = df["Date"].dropna().iloc[-1]
    years = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0:
        return None

    if use_balance:
        start_val = float(df["Account Balance"].dropna().iloc[0])
        end_val = float(df["Account Balance"].dropna().iloc[-1])
    else:
        start_val = 1.0
        end_val = float(1.0 + (df["Cumulative Percentage Returns"].dropna().iloc[-1] / 100.0))

    if start_val <= 0 or end_val <= 0:
        return None

    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def calmar_ratio(state) -> Optional[float]:
    _cagr = cagr(state, use_balance=True)
    _mdd = max_drawdown(state, use_balance=True)
    if _cagr is None or _mdd is None or _mdd == 0:
        return None
    return float(_cagr / _mdd)


def total_trades(state) -> Optional[int]:
    df = _ensure_trade_df(state.backtest_results)
    return int(df["Exit Price"].notna().sum())


# -----------------------------
# TRUE MAE from OHLC (requires entry+exit timestamps)
# -----------------------------
def mae(state) -> Optional[float]:
    """
    MAE here = mean(Max Adverse Excursion) across trades, using OHLC between entry and exit.

    For each trade:
      - LONG: MAE = (min(Low between entry..exit) - entry_price) / entry_price
      - SHORT: MAE = (entry_price - max(High between entry..exit)) / entry_price
    Returned value is a positive fraction (e.g. 0.02 = 2% adverse excursion on average).

    Requirements:
      - state.trade_log_df must include entry + exit timestamps (recommended columns: 'Entry Date', 'Exit Date')
      - state.csv_data must include Date + OHLC.

    If you don’t have entry/exit timestamps, this returns None.
    """
    trades = _ensure_trade_df(state.backtest_results)
    ohlc = _ensure_ohlc_df(state.csv_data)

    time_cols = _get_entry_exit_time_cols(trades)
    if time_cols is None:
        return None  # cannot compute MAE correctly without entry+exit times

    entry_col, exit_col = time_cols
    trades[entry_col] = pd.to_datetime(trades[entry_col], errors="coerce")
    trades[exit_col] = pd.to_datetime(trades[exit_col], errors="coerce")

    # Keep only valid completed trades with times + prices
    t = trades.loc[
        trades[entry_col].notna()
        & trades[exit_col].notna()
        & trades["Entry Price"].notna()
        & trades["Exit Price"].notna()
    ].copy()
    if t.empty:
        return None

    maes = []
    ohlc_dates = ohlc["Date"]

    for _, row in t.iterrows():
        entry_dt = row[entry_col]
        exit_dt = row[exit_col]
        if exit_dt < entry_dt:
            entry_dt, exit_dt = exit_dt, entry_dt

        entry_price = float(row["Entry Price"])
        side = str(row["Short/Long"]).lower()

        window = ohlc.loc[(ohlc_dates >= entry_dt) & (ohlc_dates <= exit_dt)]
        if window.empty or entry_price <= 0:
            continue

        if side.startswith("l"):  # long
            worst_low = float(window["Low"].min())
            mae_frac = max(0.0, (entry_price - worst_low) / entry_price)
        elif side.startswith("s"):  # short
            worst_high = float(window["High"].max())
            mae_frac = max(0.0, (worst_high - entry_price) / entry_price)
        else:
            continue

        maes.append(mae_frac)

    if not maes:
        return None
    return float(np.mean(maes))


# -----------------------------
# Metrics table + state updater
# -----------------------------
def metrics_df(state) -> pd.DataFrame:
    data = {
        "sharpe_ratio": sharpe_ratio(state),
        "sortino_ratio": sortino_ratio(state),
        "max_drawdown": max_drawdown(state),
        "calmar_ratio": calmar_ratio(state),
        "cagr": cagr(state),
        "mae": mae(state),
        "total_trades": total_trades(state),
    }
    return pd.DataFrame([{"Metric": k, "Value": v} for k, v in data.items()])


def compute_and_assign_all(state) -> None:
    """
    Populates:
      state.sharpe_ratio, state.sortino_ratio, state.max_drawdown,
      state.calmar_ratio, state.cagr, state.mae, state.total_trades,
      state.metrics_df
    """
    state.sharpe_ratio = sharpe_ratio(state)
    state.sortino_ratio = sortino_ratio(state)
    state.max_drawdown = max_drawdown(state)
    state.calmar_ratio = calmar_ratio(state)
    state.cagr = cagr(state)
    state.mae = mae(state)
    state.total_trades = total_trades(state)
    state.metrics_df = metrics_df(state)


# -----------------------------
# Minimal example usage
# -----------------------------

    # Example pseudo-state:
    # class State: pass
    # state = State()
    # state.trade_log_df = pd.read_csv("trade_log.csv")
    # state.csv_data = pd.read_csv("asset_ohlc.csv")
    # compute_and_assign_all(state)
    # print(state.metrics_df)


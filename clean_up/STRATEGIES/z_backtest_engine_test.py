"""
Swing Z-Score Strategy + Fast Optimiser (8-core)

Goals:
- Reduce trade count to ~200-500 (longer swings)
- Improve speed (NumPy precompute + Numba loop)
- Parameter search across reasonable combinations
- Output best parameters + summary + store best trade log in state.backtest_results
"""

from __future__ import annotations

import os
import time
import math
import random
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from state import AppState
from ui.statusbar import add_text_status_backtest

# Best-effort thread hints for indicator maths
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

try:
    from numba import njit
except Exception:
    njit = None


# -----------------------------
# Indicator helpers (vectorised)
# -----------------------------
def rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float64)
    n = a.size
    if window <= 0:
        return out
    if window == 1:
        return a.astype(np.float64, copy=False)
    if n < window:
        return out

    c = np.cumsum(a, dtype=np.float64)
    # sum over [i-window+1, i] for i >= window-1
    out[window - 1:] = (c[window - 1:] - np.concatenate(([0.0], c[:-window]))) / window
    return out



def rolling_std(a: np.ndarray, window: int) -> np.ndarray:
    out = np.full(a.shape, np.nan, dtype=np.float64)
    n = a.size
    if window <= 1:
        out[:] = 0.0
        return out
    if n < window:
        return out

    m = rolling_mean(a, window)
    m2 = rolling_mean(a * a, window)
    var = m2 - m * m
    var[var < 0] = 0.0
    out = np.sqrt(var)
    return out



def zscore(a: np.ndarray, window: int) -> np.ndarray:
    m = rolling_mean(a, window)
    s = rolling_std(a, window)
    s = np.where(s == 0.0, np.nan, s)
    return (a - m) / s


def rsi_ewm(a: np.ndarray, window: int = 14) -> np.ndarray:
    # pandas-style RSI with EWM smoothing
    out = np.full_like(a, np.nan, dtype=np.float64)
    if len(a) < window + 1:
        return out
    delta = np.diff(a, prepend=np.nan)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    alpha = 1.0 / window
    avg_gain = np.full_like(a, np.nan, dtype=np.float64)
    avg_loss = np.full_like(a, np.nan, dtype=np.float64)

    # seed with simple mean over first window
    start = window
    avg_gain[start] = np.nanmean(gains[1 : start + 1])
    avg_loss[start] = np.nanmean(losses[1 : start + 1])

    for i in range(start + 1, len(a)):
        avg_gain[i] = (1 - alpha) * avg_gain[i - 1] + alpha * gains[i]
        avg_loss[i] = (1 - alpha) * avg_loss[i - 1] + alpha * losses[i]

    rs = avg_gain / np.where(avg_loss == 0.0, np.nan, avg_loss)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def atr_ewm(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.nanmax(np.vstack([tr1, tr2, tr3]), axis=0)

    alpha = 2.0 / (window + 1.0)  # ewm span=window equivalent-ish
    # seed
    if len(close) < window + 1:
        return out
    out[window] = np.nanmean(tr[1 : window + 1])
    for i in range(window + 1, len(close)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * tr[i]
    return out


def bb_width(close: np.ndarray, window: int = 20, num_std: float = 2.0) -> np.ndarray:
    sma = rolling_mean(close, window)
    sd = rolling_std(close, window)
    upper = sma + num_std * sd
    lower = sma - num_std * sd
    # percentage width
    return ((upper - lower) / sma) * 100.0


# -----------------------------
# Params for swing behaviour
# -----------------------------
@dataclass(frozen=True)
class SwingParams:
    # indicator windows
    z_window: int
    rsi_window: int
    atr_window: int

    # regime filters (trend + volatility)
    ma_fast: int
    ma_slow: int
    slope_lookback: int
    slope_up: float
    slope_down: float

    # entry thresholds (bigger magnitude => fewer trades)
    z_long: float      # negative
    z_short: float     # positive
    rsi_long_max: float
    rsi_short_min: float

    # volatility regime filter: require BB width >= quantile
    bb_window: int
    bb_num_std: float
    bb_quantile: float

    # risk / trade mgmt
    atr_stop: float
    atr_target: float
    min_hold: int
    max_hold: int
    cooldown: int

    # sizing
    trade_size: float
    leverage: float


# -----------------------------
# Fast backtest core (Numba)
# -----------------------------
def _ensure_numba():
    if njit is None:
        raise ImportError("Numba not installed. Install via: pip install numba")


@njit(cache=True)
def _backtest_core(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    z: np.ndarray,
    rsi: np.ndarray,
    atr: np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
    ma_slope: np.ndarray,
    bbw: np.ndarray,
    bbw_thresh: float,
    p: Tuple,
):
    # unpack params tuple (numba-friendly)
    (
        slope_up, slope_down,
        z_long, z_short,
        rsi_long_max, rsi_short_min,
        atr_stop, atr_target,
        min_hold, max_hold,
        cooldown,
        trade_size, leverage,
    ) = p

    n = len(close)
    starting_balance = 100000.0
    bal = starting_balance

    position = 0  # 0 flat, 1 long, -1 short
    entry_price = 0.0
    entry_i = -1
    entry_atr = 0.0
    stop = 0.0
    target = 0.0

    bars_since_trade = 10_000
    last_loss = 0

    # store trades (prealloc upper bound; will trim)
    max_trades = n // 2
    entry_idx = np.empty(max_trades, dtype=np.int64)
    exit_idx = np.empty(max_trades, dtype=np.int64)
    side = np.empty(max_trades, dtype=np.int8)
    entry_px = np.empty(max_trades, dtype=np.float64)
    exit_px = np.empty(max_trades, dtype=np.float64)
    pnl_arr = np.empty(max_trades, dtype=np.float64)
    reason = np.empty(max_trades, dtype=np.int8)  # 1=meanrev 2=stop 3=tp 4=time 5=eod

    t = 0

    for i in range(n):
        c = close[i]
        h = high[i]
        l = low[i]

        if math.isnan(c) or math.isnan(z[i]) or math.isnan(rsi[i]) or math.isnan(atr[i]) or math.isnan(ma_slope[i]) or math.isnan(bbw[i]):
            bars_since_trade += 1
            continue

        bars_since_trade += 1

        is_up = ma_slope[i] > slope_up
        is_down = ma_slope[i] < slope_down

        # volatility regime filter
        if bbw[i] < bbw_thresh:
            # still allow exits if in position
            pass

        # EXIT
        if position != 0:
            exit_now = False
            exit_reason = 0
            xprice = c

            held = i - entry_i

            # stop/tp based on intrabar extremes
            if position == 1:
                if l <= stop:
                    exit_now = True
                    xprice = stop
                    exit_reason = 2
                elif h >= target:
                    exit_now = True
                    xprice = target
                    exit_reason = 3
            else:
                if h >= stop:
                    exit_now = True
                    xprice = stop
                    exit_reason = 2
                elif l <= target:
                    exit_now = True
                    xprice = target
                    exit_reason = 3

            # mean reversion exit (swing version: only after min_hold, and require stronger reversion)
            if (not exit_now) and (held >= min_hold):
                if position == 1 and z[i] >= -0.1:
                    exit_now = True
                    exit_reason = 1
                elif position == -1 and z[i] <= 0.1:
                    exit_now = True
                    exit_reason = 1

            # time exit
            if (not exit_now) and (held >= max_hold):
                exit_now = True
                exit_reason = 4

            if exit_now:
                if position == 1:
                    pct = (xprice - entry_price) / entry_price * 100.0
                else:
                    pct = (entry_price - xprice) / entry_price * 100.0

                pnl = (trade_size * leverage) * (pct / 100.0)
                bal += pnl

                last_loss = 1 if pnl < 0 else 0
                bars_since_trade = 0

                if t < max_trades:
                    entry_idx[t] = entry_i
                    exit_idx[t] = i
                    side[t] = 1 if position == 1 else -1
                    entry_px[t] = entry_price
                    exit_px[t] = xprice
                    pnl_arr[t] = pnl
                    reason[t] = exit_reason
                    t += 1

                # reset
                position = 0
                entry_price = 0.0
                entry_i = -1
                entry_atr = 0.0
                stop = 0.0
                target = 0.0

            continue

        # ENTRY (flat)
        if last_loss == 1 and bars_since_trade < cooldown:
            continue

        if bbw[i] < bbw_thresh:
            continue

        # regime: swing = only trade with clear trend OR strong countertrend extremes
        # Here: we only long if uptrend OR VERY deep oversold; only short if downtrend OR VERY deep overbought.
        long_ok = is_up or (z[i] <= (z_long - 0.7))
        short_ok = is_down or (z[i] >= (z_short + 0.7))

        # entries
        if long_ok and (z[i] <= z_long) and (rsi[i] <= rsi_long_max):
            position = 1
            entry_price = c
            entry_i = i
            entry_atr = atr[i]
            stop = entry_price - entry_atr * atr_stop
            target = entry_price + entry_atr * atr_target
            continue

        if short_ok and (z[i] >= z_short) and (rsi[i] >= rsi_short_min):
            position = -1
            entry_price = c
            entry_i = i
            entry_atr = atr[i]
            stop = entry_price + entry_atr * atr_stop
            target = entry_price - entry_atr * atr_target
            continue

    # EOD close if still open
    if position != 0 and entry_i >= 0:
        i = n - 1
        c = close[i]
        xprice = c
        if position == 1:
            pct = (xprice - entry_price) / entry_price * 100.0
        else:
            pct = (entry_price - xprice) / entry_price * 100.0
        pnl = (trade_size * leverage) * (pct / 100.0)
        bal += pnl

        if t < len(entry_idx):
            entry_idx[t] = entry_i
            exit_idx[t] = i
            side[t] = 1 if position == 1 else -1
            entry_px[t] = entry_price
            exit_px[t] = xprice
            pnl_arr[t] = pnl
            reason[t] = 5
            t += 1

    return bal, entry_idx[:t], exit_idx[:t], side[:t], entry_px[:t], exit_px[:t], pnl_arr[:t], reason[:t]


def _reason_text(code: int) -> str:
    return {
        1: "Mean Reversion",
        2: "Stop Loss",
        3: "Take Profit",
        4: "Time Exit",
        5: "End of Data",
    }.get(int(code), "Unknown")


# -----------------------------
# Precompute indicators once
# -----------------------------
def _prepare_arrays(df: pd.DataFrame, p: SwingParams):
    close = df["Close"].to_numpy(dtype=np.float64)
    high = df["High"].to_numpy(dtype=np.float64)
    low = df["Low"].to_numpy(dtype=np.float64)

    z = zscore(close, p.z_window)
    r = rsi_ewm(close, p.rsi_window)
    a = atr_ewm(high, low, close, p.atr_window)

    ma_f = rolling_mean(close, p.ma_fast)
    ma_s = rolling_mean(close, p.ma_slow)

    # slope in % over lookback
    denom = np.roll(ma_f, p.slope_lookback)
    denom[: p.slope_lookback] = np.nan
    denom = np.where(denom == 0.0, np.nan, denom)
    slope = (ma_f - np.roll(ma_f, p.slope_lookback)) / denom * 100.0
    slope[: p.slope_lookback] = np.nan

    bbw = bb_width(close, p.bb_window, p.bb_num_std)

    return close, high, low, z, r, a, ma_f, ma_s, slope, bbw


def _bbw_threshold(bbw: np.ndarray, q: float) -> float:
    x = bbw[~np.isnan(bbw)]
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


# -----------------------------
# Optimisation (multiprocessing)
# -----------------------------
def _score_run(final_balance: float, pnls: np.ndarray, trade_count: int) -> float:
    # maximise return, lightly penalise extreme trade counts outside target
    ret = (final_balance - 100000.0) / 100000.0
    if trade_count == 0:
        return -1e9
    # small penalty for very low/high count
    penalty = 0.0
    if trade_count < 200:
        penalty = (200 - trade_count) * 0.0005
    elif trade_count > 500:
        penalty = (trade_count - 500) * 0.0002

    # drawdown proxy using cumulative pnl
    eq = 100000.0 + np.cumsum(pnls)
    peak = np.maximum.accumulate(eq)
    dd = np.max((peak - eq) / np.where(peak == 0, 1.0, peak))
    dd_pen = dd * 0.5  # weight drawdown

    return ret - penalty - dd_pen


def _worker_eval(args):
    # unpack
    df_np_pack, dates, params = args
    (close, high, low, z, rsi, atr, ma_f, ma_s, slope, bbw) = df_np_pack
    # bbw threshold based on params quantile
    bbw_th = _bbw_threshold(bbw, params.bb_quantile)

    # numba params tuple
    ptuple = (
        params.slope_up, params.slope_down,
        params.z_long, params.z_short,
        params.rsi_long_max, params.rsi_short_min,
        params.atr_stop, params.atr_target,
        params.min_hold, params.max_hold,
        params.cooldown,
        params.trade_size, params.leverage,
    )

    final_bal, e_i, x_i, side, e_px, x_px, pnls, reasons = _backtest_core(
        close, high, low, z, rsi, atr, ma_f, ma_s, slope, bbw, bbw_th, ptuple
    )
    trades = int(pnls.size)
    score = _score_run(final_bal, pnls, trades)

    return score, float(final_bal), trades, params, (e_i, x_i, side, e_px, x_px, pnls, reasons)


def _sample_params(rng: random.Random) -> SwingParams:
    # "reasonable combinations" for swing behaviour
    z_window = rng.choice([30, 40, 50, 60])
    rsi_window = rng.choice([14, 21])
    atr_window = rng.choice([14, 21])

    ma_fast = rng.choice([50, 75, 100])
    ma_slow = rng.choice([150, 200, 250])
    slope_lookback = rng.choice([20, 30, 40])

    slope_up = rng.choice([0.25, 0.35, 0.5, 0.7])
    slope_down = -slope_up

    # stronger thresholds = fewer trades
    z_long = rng.choice([-2.2, -2.5, -2.8, -3.0])
    z_short = rng.choice([2.2, 2.5, 2.8, 3.0])

    rsi_long_max = rng.choice([45, 50, 55])
    rsi_short_min = rng.choice([50, 55, 60])

    bb_window = rng.choice([20, 30])
    bb_num_std = rng.choice([2.0, 2.2])
    bb_quantile = rng.choice([0.35, 0.45, 0.55])  # require higher vol than scalping version

    atr_stop = rng.choice([2.5, 3.0, 3.5])
    atr_target = rng.choice([5.0, 6.0, 7.0])  # bigger targets for swings

    min_hold = rng.choice([5, 10, 15])
    max_hold = rng.choice([120, 160, 200])    # longer holds
    cooldown = rng.choice([5, 8, 12])         # reduce churn

    trade_size = 20000.0
    leverage = 10.0

    return SwingParams(
        z_window=z_window,
        rsi_window=rsi_window,
        atr_window=atr_window,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        slope_lookback=slope_lookback,
        slope_up=float(slope_up),
        slope_down=float(slope_down),
        z_long=float(z_long),
        z_short=float(z_short),
        rsi_long_max=float(rsi_long_max),
        rsi_short_min=float(rsi_short_min),
        bb_window=bb_window,
        bb_num_std=float(bb_num_std),
        bb_quantile=float(bb_quantile),
        atr_stop=float(atr_stop),
        atr_target=float(atr_target),
        min_hold=int(min_hold),
        max_hold=int(max_hold),
        cooldown=int(cooldown),
        trade_size=float(trade_size),
        leverage=float(leverage),
    )


# -----------------------------
# Public entry: optimise + run best
# -----------------------------
def zscore_swing_optimised(state: AppState) -> pd.DataFrame:
    _ensure_numba()

    if not hasattr(state, "csv_data") or state.csv_data is None:
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    df = state.csv_data.copy()

    if "Close" not in df.columns:
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # synthetic OHLC if missing
    if "Open" not in df.columns:
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    else:
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")

    if "High" not in df.columns:
        df["High"] = df["Close"]
    else:
        df["High"] = pd.to_numeric(df["High"], errors="coerce")

    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    else:
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

    df = df.dropna(subset=["Close", "High", "Low"]).reset_index(drop=True)

    if len(df) < 600:
        # swing strategy needs a bit more history to avoid warmup dominating
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    dates = df["Date"].to_numpy() if "Date" in df.columns else np.array([None] * len(df), dtype=object)

    # fixed seed for reproducibility
    rng = random.Random(1337)

    # --- choose search budget ---
    # “reasonable combination of variables” = random search over curated ranges.
    # Increase iterations if you want stronger results.
    iterations = 240  # ~240 trials, parallel over 8 cores

    # Precompute arrays ONCE using a "max" config? We actually need per-param windows.
    # To keep it fast and simple, we compute per-trial. Still very quick with Numba core + small trials.
    # If you want even more speed, we can precompute superset indicators and slice, but that’s more code.

    # Build tasks
    tasks = []
    for _ in range(iterations):
        p = _sample_params(rng)
        # Prepare arrays per-param (fast enough for this budget)
        df_np_pack = _prepare_arrays(df, p)
        tasks.append((df_np_pack, dates, p))

    # Evaluate on 8 cores
    cores = min(8, mp.cpu_count())
    t0 = time.time()
    with mp.Pool(processes=cores) as pool:
        results = pool.map(_worker_eval, tasks)
    t1 = time.time()

    # Filter to target trade counts, then pick best score
    filtered = [r for r in results if 200 <= r[2] <= 500]
    if not filtered:
        # fallback: pick best overall if nothing hits the trade window
        filtered = results

    best = max(filtered, key=lambda x: x[0])
    best_score, best_final_bal, best_trades, best_params, best_trade_pack = best

    e_i, x_i, side, e_px, x_px, pnls, reasons = best_trade_pack

    # Build trade log DF
    trade_dates = []
    for idx in x_i:
        trade_dates.append(dates[int(idx)] if len(dates) else None)

    cum_ret = (100000.0 + np.cumsum(pnls) - 100000.0) / 100000.0 * 100.0
    bal_hist = 100000.0 + np.cumsum(pnls)

    out = pd.DataFrame(
        {
            "Date": trade_dates,
            "Cumulative Percentage Returns": cum_ret.astype(float),
            "Account Balance": bal_hist.astype(float),
            "Short/Long": np.where(side == 1, "Long", "Short"),
            "Entry Price": e_px.astype(float),
            "Exit Price": x_px.astype(float),
            "PnL": pnls.astype(float),
            "Exit Reason": [_reason_text(int(r)) for r in reasons],
            "Entry Index": e_i.astype(int),
            "Exit Index": x_i.astype(int),
        }
    )

    if "Date" in out.columns and not out["Date"].isnull().all():
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    state.backtest_results = out

    # Summary stats
    total_pnl = float(np.nansum(pnls)) if pnls.size else 0.0
    wins = int(np.sum(pnls > 0))
    losses = int(np.sum(pnls < 0))
    win_rate = (wins / max(1, int(pnls.size))) * 100.0

    # drawdown
    eq = 100000.0 + np.cumsum(pnls)
    peak = np.maximum.accumulate(eq) if eq.size else np.array([100000.0])
    dd = float(np.max((peak - eq) / np.where(peak == 0, 1.0, peak))) if eq.size else 0.0

    add_text_status_backtest(state, "=" * 60)
    add_text_status_backtest(state, "SWING Z-SCORE OPTIMISER RESULTS")
    add_text_status_backtest(state, "=" * 60)
    add_text_status_backtest(state, f"Trials: {iterations} | Cores: {cores} | Time: {t1 - t0:.2f}s")
    add_text_status_backtest(state, f"Best Score: {best_score:.6f}")
    add_text_status_backtest(state, f"Trades: {best_trades} | Win Rate: {win_rate:.1f}% | Max DD: {dd*100:.2f}%")
    add_text_status_backtest(state, f"Total PnL: ${total_pnl:+.2f} | Final Balance: ${best_final_bal:.2f}")
    add_text_status_backtest(state, "-" * 60)
    add_text_status_backtest(state, "Best Parameters:")
    for k, v in asdict(best_params).items():
        add_text_status_backtest(state, f"  {k}: {v}")
    add_text_status_backtest(state, "=" * 60)

    return out


# Export names for strategy registration
strategy_name = "zscore_swing_optimised"
strategy_entry = zscore_swing_optimised

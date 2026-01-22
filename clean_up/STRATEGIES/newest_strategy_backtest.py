from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

# =============================
# Indicator helpers (fast enough)
# =============================
def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(span=window, min_periods=window, adjust=False).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=close.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm, index=close.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx


def safe_pct_move(entry: float, exit_: float, side: int) -> float:
    if entry <= 0 or exit_ <= 0:
        return 0.0
    if side == 1:
        return (exit_ - entry) / entry * 100.0
    return (entry - exit_) / entry * 100.0


# ==========================================
# 4h regime from 15m (built once per dataset)
# ==========================================
def build_4h_regime_from_15m(df15: pd.DataFrame) -> pd.DataFrame:
    df = df15.dropna(subset=["Date"]).copy()
    df = df.set_index("Date")

    o = df["Open"].resample("4H").first()
    h = df["High"].resample("4H").max()
    l = df["Low"].resample("4H").min()
    c = df["Close"].resample("4H").last()

    df4 = pd.DataFrame({"Date": o.index, "Open": o.values, "High": h.values, "Low": l.values, "Close": c.values})
    df4 = df4.dropna().reset_index(drop=True)

    df4["ema_50"] = calc_ema(df4["Close"], span=50)
    denom = df4["ema_50"].shift(3).replace(0, np.nan)
    df4["ema_50_slope_pct"] = ((df4["ema_50"] - df4["ema_50"].shift(3)) / denom) * 100.0
    df4["adx"] = calc_adx(df4["High"], df4["Low"], df4["Close"], window=14)
    return df4


def attach_4h_regime_to_15m(df15: pd.DataFrame, df4: pd.DataFrame) -> pd.DataFrame:
    df15_sorted = df15.sort_values("Date").copy()
    df4_sorted = df4.sort_values("Date").copy()

    merged = pd.merge_asof(
        df15_sorted,
        df4_sorted[["Date", "adx", "ema_50_slope_pct"]],
        on="Date",
        direction="backward",
    )
    return merged


# =============================
# Params to sweep
# =============================
@dataclass(frozen=True)
class SweepParams:
    adx_trend_threshold: float
    slope_trend_threshold: float
    trend_z_entry: float
    range_z_extreme: float
    trend_rsi_reclaim: float
    k_sl: float
    k_tp: float
    tp_floor_base: float  # e.g. 0.0030 .. 0.0060
    max_hold_trend: int
    max_hold_range: int


# Global dataset per worker (avoid re-pickling big DF each task)
_GDATA: Optional[pd.DataFrame] = None


def _init_worker(df_prepared: pd.DataFrame):
    global _GDATA
    _GDATA = df_prepared


# ==========================================
# Single backtest run (worker)
# ==========================================
def _run_one(params: SweepParams) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """
    Returns:
      score (float): primary objective (Total PnL)
      summary (dict)
      results_df (pd.DataFrame) - trade log in your exact schema
    """
    df = _GDATA
    if df is None or df.empty:
        return -1e18, {"error": "No data"}, pd.DataFrame()

    # --- Costs (kept low, but non-zero) ---
    # Tune if you want. This is deliberately small as you said.
    fee_rate = 0.00015       # 0.015% per side
    slippage_rate = 0.00005  # 0.005% per side
    per_side_cost = fee_rate + slippage_rate
    round_trip_cost = 2.0 * per_side_cost

    tp_floor = params.tp_floor_base + round_trip_cost

    starting_balance = 100000.0
    trade_size = 20000.0
    leverage = 10.0

    account_balance = starting_balance

    position = 0
    entry_price: Optional[float] = None
    entry_idx: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_regime: Optional[str] = None

    bars_since_last_trade = 999
    last_trade_was_loss = False
    cooldown_bars = 4

    trade_dates: List[Any] = []
    cumulative_pct_returns: List[float] = []
    account_balance_history: List[float] = []
    trade_sides: List[str] = []
    trade_entry_prices: List[float] = []
    trade_exit_prices: List[float] = []
    trade_pnl: List[float] = []
    trade_exit_reasons: List[str] = []

    length = len(df)

    for i in range(length):
        close_ = df.iat[i, df.columns.get_loc("Close")]
        high_ = df.iat[i, df.columns.get_loc("High")]
        low_ = df.iat[i, df.columns.get_loc("Low")]
        rsi_ = df.iat[i, df.columns.get_loc("rsi")]
        atrp_ = df.iat[i, df.columns.get_loc("atrp")]
        resid_z_ = df.iat[i, df.columns.get_loc("resid_z")]
        adx4_ = df.iat[i, df.columns.get_loc("adx")]
        slope4_ = df.iat[i, df.columns.get_loc("ema_50_slope_pct")]

        if (
            pd.isna(close_) or pd.isna(high_) or pd.isna(low_) or
            pd.isna(rsi_) or pd.isna(atrp_) or pd.isna(resid_z_) or
            pd.isna(adx4_) or pd.isna(slope4_)
        ):
            bars_since_last_trade += 1
            continue

        bars_since_last_trade += 1

        trend_regime = (adx4_ >= params.adx_trend_threshold) and (abs(slope4_) >= params.slope_trend_threshold)
        regime = "TREND" if trend_regime else "RANGE"
        trend_dir = 1 if slope4_ > 0 else (-1 if slope4_ < 0 else 0)

        # ----------------
        # EXIT
        # ----------------
        if position != 0:
            exit_triggered = False
            exit_reason = ""
            exit_price = float(close_)

            # Stop
            if stop_loss is not None:
                if position == 1 and low_ <= stop_loss:
                    exit_triggered = True
                    exit_price = float(stop_loss)
                    exit_reason = "Stop Loss"
                elif position == -1 and high_ >= stop_loss:
                    exit_triggered = True
                    exit_price = float(stop_loss)
                    exit_reason = "Stop Loss"

            # Take profit
            if not exit_triggered and take_profit is not None:
                if position == 1 and high_ >= take_profit:
                    exit_triggered = True
                    exit_price = float(take_profit)
                    exit_reason = "Take Profit"
                elif position == -1 and low_ <= take_profit:
                    exit_triggered = True
                    exit_price = float(take_profit)
                    exit_reason = "Take Profit"

            # Mean reversion (range)
            if not exit_triggered and entry_regime == "RANGE":
                if abs(resid_z_) <= 0.25:
                    exit_triggered = True
                    exit_reason = "Mean Reversion"

            # Time stop
            if not exit_triggered and entry_idx is not None:
                max_hold = params.max_hold_trend if entry_regime == "TREND" else params.max_hold_range
                if (i - entry_idx) >= max_hold:
                    exit_triggered = True
                    exit_reason = "Time Exit"

            if exit_triggered and entry_price is not None:
                if position == 1:
                    effective_exit = exit_price * (1.0 - per_side_cost)
                else:
                    effective_exit = exit_price * (1.0 + per_side_cost)

                pct_move = safe_pct_move(entry_price, effective_exit, position)
                pnl = (trade_size * leverage) * (pct_move / 100.0)
                account_balance += pnl

                last_trade_was_loss = pnl < 0
                bars_since_last_trade = 0

                dt = df.iat[i, df.columns.get_loc("Date")] if "Date" in df.columns else None
                trade_dates.append(dt)
                cumulative_pct_returns.append((account_balance - starting_balance) / starting_balance * 100.0)
                account_balance_history.append(account_balance)
                trade_sides.append("Long" if position == 1 else "Short")
                trade_entry_prices.append(entry_price)
                trade_exit_prices.append(effective_exit)
                trade_pnl.append(pnl)
                trade_exit_reasons.append(exit_reason)

                # reset
                position = 0
                entry_price = None
                entry_idx = None
                stop_loss = None
                take_profit = None
                entry_regime = None

            continue

        # ----------------
        # ENTRY
        # ----------------
        if last_trade_was_loss and bars_since_last_trade < cooldown_bars:
            continue

        if account_balance < trade_size:
            continue

        # vol floor (avoid dead chop)
        if atrp_ < 0.0006:
            continue

        sl_pct = max(0.0015, params.k_sl * float(atrp_))
        tp_pct = max(tp_floor, params.k_tp * float(atrp_))

        long_signal = False
        short_signal = False

        # TREND pullback
        if regime == "TREND" and trend_dir != 0:
            if trend_dir == 1:
                pullback_ok = (resid_z_ <= -params.trend_z_entry)
                rsi_trigger = (rsi_ >= params.trend_rsi_reclaim)
                if pullback_ok and rsi_trigger:
                    long_signal = True
            else:
                pullback_ok = (resid_z_ >= params.trend_z_entry)
                rsi_trigger = (rsi_ <= (100.0 - params.trend_rsi_reclaim))
                if pullback_ok and rsi_trigger:
                    short_signal = True

        # RANGE mean reversion
        if regime == "RANGE" and not (long_signal or short_signal):
            if resid_z_ <= -params.range_z_extreme and rsi_ <= 35.0:
                long_signal = True
            elif resid_z_ >= params.range_z_extreme and rsi_ >= 65.0:
                short_signal = True

        # Execute entries (apply per-side cost)
        if long_signal:
            position = 1
            raw_entry = float(close_)
            entry_price = raw_entry * (1.0 + per_side_cost)
            entry_idx = i
            entry_regime = regime
            stop_loss = entry_price * (1.0 - sl_pct)
            take_profit = entry_price * (1.0 + tp_pct)

        elif short_signal:
            position = -1
            raw_entry = float(close_)
            entry_price = raw_entry * (1.0 - per_side_cost)
            entry_idx = i
            entry_regime = regime
            stop_loss = entry_price * (1.0 + sl_pct)
            take_profit = entry_price * (1.0 - tp_pct)

    # Close any open position at end
    if position != 0 and entry_price is not None:
        exit_idx = length - 1
        exit_price = float(df.iat[exit_idx, df.columns.get_loc("Close")])
        if position == 1:
            effective_exit = exit_price * (1.0 - per_side_cost)
        else:
            effective_exit = exit_price * (1.0 + per_side_cost)

        pct_move = safe_pct_move(entry_price, effective_exit, position)
        pnl = (trade_size * leverage) * (pct_move / 100.0)
        account_balance += pnl

        dt = df.iat[exit_idx, df.columns.get_loc("Date")] if "Date" in df.columns else None
        trade_dates.append(dt)
        cumulative_pct_returns.append((account_balance - starting_balance) / starting_balance * 100.0)
        account_balance_history.append(account_balance)
        trade_sides.append("Long" if position == 1 else "Short")
        trade_entry_prices.append(entry_price)
        trade_exit_prices.append(effective_exit)
        trade_pnl.append(pnl)
        trade_exit_reasons.append("End of Data")

    results_df = pd.DataFrame(
        {
            "Date": trade_dates,
            "Cumulative Percentage Returns": cumulative_pct_returns,
            "Account Balance": account_balance_history,
            "Short/Long": trade_sides,
            "Entry Price": trade_entry_prices,
            "Exit Price": trade_exit_prices,
            "PnL": trade_pnl,
            "Exit Reason": trade_exit_reasons,
        }
    )

    total_pnl = float(np.nansum(trade_pnl)) if trade_pnl else 0.0
    total_trades = len(trade_pnl)
    win_rate = (sum(1 for p in trade_pnl if p > 0) / total_trades * 100.0) if total_trades else 0.0
    max_dd = _max_drawdown_from_equity(np.array(account_balance_history, dtype=float)) if account_balance_history else 0.0

    summary = {
        "total_pnl": total_pnl,
        "final_balance": account_balance,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "tp_floor_pct": tp_floor * 100.0,
        "params": params,
    }

    # Primary objective: total_pnl (you can change to Calmar/Sharpe later)
    score = total_pnl
    return score, summary, results_df


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    # returns fraction (0.25 = 25% max DD)
    if equity.size < 2:
        return 0.0
    peak = -np.inf
    max_dd = 0.0
    for x in equity:
        if not np.isfinite(x):
            continue
        peak = max(peak, x)
        if peak > 0:
            dd = (peak - x) / peak
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


# ==========================================
# Public function: called from your app
# ==========================================
def run_eth_swing_param_sweep(state) -> pd.DataFrame:
    """
    Runs param sweep on 8 cores, prints every 8 completed combos,
    stores best into state.backtest_results, saves best equity curve to disk.
    """
    if not hasattr(state, "csv_data") or state.csv_data is None:
        state.backtest_results = pd.DataFrame()
        print("ERROR: No CSV data loaded.")
        return state.backtest_results

    df = state.csv_data.copy()

    # --- Validate & coerce ---
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.reset_index(drop=True)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
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

    if "Date" not in df.columns:
        # still allow, but 4h resample will fail - so enforce it
        print("ERROR: Your data must have a Date column for 4H regime building.")
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    if df["Date"].isna().all():
        print("ERROR: Date column could not be parsed.")
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    # --- Precompute indicators ONCE (huge speedup) ---
    df["fair"] = calc_ema(df["Close"], span=50)
    df["residual"] = df["Close"] - df["fair"]
    resid_std = df["residual"].rolling(window=80, min_periods=80).std().replace(0, np.nan)
    df["resid_z"] = df["residual"] / resid_std

    df["rsi"] = calc_rsi(df["Close"], window=14)
    df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], window=14)
    df["atrp"] = df["atr"] / df["Close"].replace(0, np.nan)

    df4 = build_4h_regime_from_15m(df)
    df = attach_4h_regime_to_15m(df, df4)

    # drop rows where core indicators not ready
    df = df.dropna(subset=["Close", "High", "Low", "rsi", "atrp", "resid_z", "adx", "ema_50_slope_pct"]).reset_index(drop=True)
    if len(df) < 800:
        print(f"ERROR: Not enough clean indicator-ready rows after preprocessing: {len(df)}")
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    # --- Build param grid (edit these ranges as you like) ---
    grid: List[SweepParams] = []
    for adx_th in (20.0, 25.0, 30.0):
        for slope_th in (0.10, 0.20, 0.35):
            for trend_z in (0.8, 1.0, 1.2):
                for range_z in (1.8, 2.1, 2.4):
                    for rsi_rec in (46.0, 48.0, 50.0):
                        for k_sl in (0.9, 1.2, 1.5):
                            for k_tp in (0.8, 1.0, 1.2):
                                for tp_floor in (0.0030, 0.0045, 0.0060):
                                    grid.append(
                                        SweepParams(
                                            adx_trend_threshold=adx_th,
                                            slope_trend_threshold=slope_th,
                                            trend_z_entry=trend_z,
                                            range_z_extreme=range_z,
                                            trend_rsi_reclaim=rsi_rec,
                                            k_sl=k_sl,
                                            k_tp=k_tp,
                                            tp_floor_base=tp_floor,
                                            max_hold_trend=48,
                                            max_hold_range=24,
                                        )
                                    )

    print(f"Running sweep: {len(grid)} combinations on 8 cores...")

    best_score = -1e18
    best_summary: Dict[str, Any] = {}
    best_df = pd.DataFrame()

    done = 0
    batch = 0

    # --- Parallel run ---
    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker, initargs=(df,)) as ex:
        futures = [ex.submit(_run_one, p) for p in grid]

        for fut in as_completed(futures):
            score, summary, res_df = fut.result()
            done += 1

            if score > best_score:
                best_score = score
                best_summary = summary
                best_df = res_df

            if done % 8 == 0:
                batch += 1
                print(f"[Batch {batch}] Completed {done}/{len(grid)} | Current best Total PnL: ${best_score:+.2f}")

    # --- Print best params ---
    if not best_summary:
        print("No valid results.")
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    p = best_summary["params"]
    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"Total PnL:      ${best_summary['total_pnl']:+.2f}")
    print(f"Final Balance:  ${best_summary['final_balance']:.2f}")
    print(f"Trades:         {best_summary['total_trades']}")
    print(f"Win Rate:       {best_summary['win_rate']:.1f}%")
    print(f"Max Drawdown:   {best_summary['max_drawdown']*100:.2f}%")
    print(f"TP Floor:       {best_summary['tp_floor_pct']:.3f}%")
    print("-" * 60)
    print("Params:")
    print(p)
    print("=" * 60 + "\n")

    # Save best equity curve (your trade-log DF already includes balance curve)
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "best_equity_curve.csv")
    best_df.to_csv(csv_path, index=False)
    print(f"Saved best equity curve CSV -> {csv_path}")

    # Save PNG plot
    _save_equity_plot(best_df, os.path.join(out_dir, "best_equity_curve.png"))

    # Put best into state so your metrics pipeline can run on it
    state.backtest_results = best_df
    return best_df


def _save_equity_plot(results_df: pd.DataFrame, path: str) -> None:
    try:
        import matplotlib.pyplot as plt

        if results_df.empty or "Account Balance" not in results_df.columns:
            print("Plot skipped: no Account Balance column.")
            return

        y = pd.to_numeric(results_df["Account Balance"], errors="coerce").dropna().values
        if y.size < 2:
            print("Plot skipped: not enough points.")
            return

        plt.figure()
        plt.plot(y)
        plt.title("Best Equity Curve (Account Balance)")
        plt.xlabel("Trade #")
        plt.ylabel("Balance")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved best equity curve PNG -> {path}")
    except Exception as e:
        print(f"Plot failed: {e}")


# Export names for your strategy registration
strategy_name = "eth_swing_param_sweep"
strategy_entry = run_eth_swing_param_sweep

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List

from state import AppState

# Reuse model loader + indicator helpers from ml script
from AI.ml_xgb_filter import (
    load_xgb_filter_model,
    calc_rsi,
    calc_atr,
    calc_ema,
    calc_adx,
    build_4h_regime_from_15m,
    attach_4h_regime_to_15m,
)

# -----------------------------
# Utility
# -----------------------------
def safe_pct_move(entry: float, exit_: float, side: int) -> float:
    if entry <= 0 or exit_ <= 0:
        return 0.0
    if side == 1:
        return (exit_ - entry) / entry * 100.0
    return (entry - exit_) / entry * 100.0


def _compute_ml_features_at_index(df: pd.DataFrame, i: int) -> pd.DataFrame:
    """
    Must match feature engineering used during training (same columns).
    Returns a 1-row DataFrame.
    """
    row = {}

    close_ = float(df.at[i, "Close"])
    fair_ = float(df.at[i, "fair"])
    o = float(df.at[i, "Open"])
    h = float(df.at[i, "High"])
    l = float(df.at[i, "Low"])
    c = close_

    rng = max(h - l, 1e-9)

    row["resid_z"] = float(df.at[i, "resid_z"])
    row["rsi"] = float(df.at[i, "rsi"])
    row["atrp"] = float(df.at[i, "atrp"])
    row["adx4"] = float(df.at[i, "adx"])
    row["slope4"] = float(df.at[i, "ema_50_slope_pct"])

    row["close"] = close_
    row["fair"] = fair_
    row["dist_fair_pct"] = (close_ - fair_) / (fair_ if fair_ != 0 else 1e-9)

    row["candle_body_pct"] = (c - o) / (o if o != 0 else 1e-9)
    row["candle_range_pct"] = (rng) / (o if o != 0 else 1e-9)
    row["upper_wick_frac"] = (h - max(o, c)) / rng
    row["lower_wick_frac"] = (min(o, c) - l) / rng

    # recent returns
    # (safe: may be NaN early; caller should skip if NaNs)
    row["ret_1"] = float(df["Close"].pct_change(1).iat[i])
    row["ret_4"] = float(df["Close"].pct_change(4).iat[i])
    row["ret_12"] = float(df["Close"].pct_change(12).iat[i])
    row["ret_48"] = float(df["Close"].pct_change(48).iat[i])

    # side + regime are injected by caller
    row["side"] = 0
    row["is_trend_regime"] = 0
    row["trend_dir"] = 0

    dt = pd.to_datetime(df.at[i, "Date"])
    row["hour"] = int(dt.hour)
    row["dow"] = int(dt.dayofweek)

    return pd.DataFrame([row])


def _ml_scale(p: float, threshold: float, min_scale: float, max_scale: float) -> float:
    """
    Piecewise linear scaling:
      p <= threshold -> 0 (gated)
      p == threshold -> min_scale
      p == 1 -> max_scale
    """
    if p <= threshold:
        return 0.0
    t = (p - threshold) / max(1e-9, (1.0 - threshold))
    scale = min_scale + t * (max_scale - min_scale)
    return float(np.clip(scale, min_scale, max_scale))


# -----------------------------
# Strategy (best params + ML filter)
# -----------------------------
def eth_swing_regime_strategy_best_ml(state: AppState) -> pd.DataFrame:
    # ---- Load model (once) ----
    if getattr(state, "ml_enabled", True):
        if not hasattr(state, "ml_model") or state.ml_model is None:
            load_xgb_filter_model(state)
        clf = state.ml_model
        feature_names: List[str] = state.ml_feature_names
    else:
        clf = None
        feature_names = []

    # ---- Load & validate data ----
    if not hasattr(state, "csv_data") or state.csv_data is None:
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    df = state.csv_data.copy()

    if "Close" not in df.columns or "Date" not in df.columns:
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

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

    if len(df) < 800:
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    # ---- Indicators (same as best strategy) ----
    df["fair"] = calc_ema(df["Close"], span=50)
    df["residual"] = df["Close"] - df["fair"]
    resid_std = df["residual"].rolling(window=80, min_periods=80).std().replace(0, np.nan)
    df["resid_z"] = df["residual"] / resid_std

    df["rsi"] = calc_rsi(df["Close"], window=14)
    df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], window=14)
    df["atrp"] = df["atr"] / df["Close"].replace(0, np.nan)

    df4 = build_4h_regime_from_15m(df)
    df = attach_4h_regime_to_15m(df, df4)

    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "rsi", "atrp", "resid_z", "adx", "ema_50_slope_pct"]).reset_index(drop=True)
    if len(df) < 800:
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    length = len(df)

    # ---- Best locked params ----
    adx_trend_threshold = 20.0
    slope_trend_threshold = 0.1
    trend_z_entry = 1.0
    range_z_extreme = 2.4
    trend_rsi_reclaim = 46.0
    k_sl = 0.9
    k_tp = 0.8
    tp_floor_base = 0.006
    max_holding_bars_trend = 48
    max_holding_bars_range = 24
    cooldown_bars = 4

    # ---- Low costs ----
    fee_rate = 0.00015
    slippage_rate = 0.00005
    per_side_cost = fee_rate + slippage_rate
    round_trip_cost = 2.0 * per_side_cost
    tp_floor = tp_floor_base + round_trip_cost

    # ---- Account / sizing ----
    starting_balance = 100000.0
    base_trade_size = 20000.0
    leverage = 10.0
    account_balance = starting_balance

    # ML parameters from state
    ml_threshold = float(getattr(state, "ml_threshold", 0.60))
    ml_min_scale = float(getattr(state, "ml_min_scale", 0.35))
    ml_max_scale = float(getattr(state, "ml_max_scale", 1.00))

    # ---- Position state ----
    position = 0
    entry_price: Optional[float] = None
    entry_idx: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_regime: Optional[str] = None
    trade_size: float = base_trade_size  # per-trade (after ML scaling)

    bars_since_last_trade = 999
    last_trade_was_loss = False

    # ---- Output ----
    trade_dates = []
    cumulative_pct_returns = []
    account_balance_history = []
    trade_sides = []
    trade_entry_prices = []
    trade_exit_prices = []
    trade_pnl = []
    trade_exit_reasons = []

    for i in range(length):
        close_ = float(df.at[i, "Close"])
        high_ = float(df.at[i, "High"])
        low_ = float(df.at[i, "Low"])
        rsi_ = float(df.at[i, "rsi"])
        atrp_ = float(df.at[i, "atrp"])
        resid_z_ = float(df.at[i, "resid_z"])
        adx4_ = float(df.at[i, "adx"])
        slope4_ = float(df.at[i, "ema_50_slope_pct"])

        bars_since_last_trade += 1

        trend_regime = (adx4_ >= adx_trend_threshold) and (abs(slope4_) >= slope_trend_threshold)
        regime = "TREND" if trend_regime else "RANGE"
        trend_dir = 1 if slope4_ > 0 else (-1 if slope4_ < 0 else 0)

        # =========================
        # Exit
        # =========================
        if position != 0:
            exit_triggered = False
            exit_reason = ""
            exit_price = close_

            # Stop loss
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

            # Range mean reversion exit
            if not exit_triggered and entry_regime == "RANGE":
                if abs(resid_z_) <= 0.25:
                    exit_triggered = True
                    exit_reason = "Mean Reversion"

            # Time exit
            if not exit_triggered and entry_idx is not None:
                max_hold = max_holding_bars_trend if entry_regime == "TREND" else max_holding_bars_range
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

                trade_date = df.at[i, "Date"]
                trade_dates.append(trade_date)
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
                trade_size = base_trade_size

            continue

        # =========================
        # Entry
        # =========================
        if last_trade_was_loss and bars_since_last_trade < cooldown_bars:
            continue
        if account_balance < base_trade_size:
            continue
        if atrp_ < 0.0006:
            continue

        sl_pct = max(0.0015, k_sl * atrp_)
        tp_pct = max(tp_floor, k_tp * atrp_)

        long_signal = False
        short_signal = False

        # Base signal logic (same as best)
        if regime == "TREND" and trend_dir != 0:
            if trend_dir == 1:
                if (resid_z_ <= -trend_z_entry) and (rsi_ >= trend_rsi_reclaim):
                    long_signal = True
            else:
                if (resid_z_ >= trend_z_entry) and (rsi_ <= (100.0 - trend_rsi_reclaim)):
                    short_signal = True

        if regime == "RANGE" and not (long_signal or short_signal):
            if resid_z_ <= -range_z_extreme and rsi_ <= 35.0:
                long_signal = True
            elif resid_z_ >= range_z_extreme and rsi_ >= 65.0:
                short_signal = True

        if not (long_signal or short_signal):
            continue

        side = 1 if long_signal else -1

        # ---- ML gate + scale ----
        if clf is not None:
            feats = _compute_ml_features_at_index(df, i)

            # inject side/regime/trend_dir to match training
            feats.at[0, "side"] = int(side)
            feats.at[0, "is_trend_regime"] = 1 if regime == "TREND" else 0
            feats.at[0, "trend_dir"] = 1 if slope4_ > 0 else (-1 if slope4_ < 0 else 0)

            # any NaNs => skip (prevents garbage inference)
            if feats.replace([np.inf, -np.inf], np.nan).isna().any(axis=None):
                continue

            # reorder to model feature list
            feats = feats[feature_names]

            p = float(clf.predict_proba(feats)[:, 1][0])
            scale = _ml_scale(p, ml_threshold, ml_min_scale, ml_max_scale)
            if scale <= 0.0:
                continue

            trade_size = base_trade_size * scale
        else:
            trade_size = base_trade_size

        # Execute entry (apply entry cost)
        if side == 1:
            position = 1
            raw_entry = close_
            entry_price = raw_entry * (1.0 + per_side_cost)
            entry_idx = i
            entry_regime = regime
            stop_loss = entry_price * (1.0 - sl_pct)
            take_profit = entry_price * (1.0 + tp_pct)
        else:
            position = -1
            raw_entry = close_
            entry_price = raw_entry * (1.0 - per_side_cost)
            entry_idx = i
            entry_regime = regime
            stop_loss = entry_price * (1.0 + sl_pct)
            take_profit = entry_price * (1.0 - tp_pct)

    # Close any open position at end
    if position != 0 and entry_price is not None:
        exit_idx = length - 1
        exit_price = float(df.at[exit_idx, "Close"])
        if position == 1:
            effective_exit = exit_price * (1.0 - per_side_cost)
        else:
            effective_exit = exit_price * (1.0 + per_side_cost)

        pct_move = safe_pct_move(entry_price, effective_exit, position)
        pnl = (trade_size * leverage) * (pct_move / 100.0)
        account_balance += pnl

        trade_date = df.at[exit_idx, "Date"]
        trade_dates.append(trade_date)
        cumulative_pct_returns.append((account_balance - starting_balance) / starting_balance * 100.0)
        account_balance_history.append(account_balance)
        trade_sides.append("Long" if position == 1 else "Short")
        trade_entry_prices.append(entry_price)
        trade_exit_prices.append(effective_exit)
        trade_pnl.append(pnl)
        trade_exit_reasons.append("End of Data")

    # Results schema (unchanged)
    final_results_df = pd.DataFrame(
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

    if "Date" in final_results_df.columns and not final_results_df["Date"].isnull().all():
        final_results_df["Date"] = pd.to_datetime(final_results_df["Date"], errors="coerce")

    state.backtest_results = final_results_df
    return final_results_df


# Export names for strategy registration
strategy_name = "eth_swing_best_with_ml_filter"
strategy_entry = eth_swing_regime_strategy_best_ml

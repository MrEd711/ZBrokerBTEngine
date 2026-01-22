"""
Advanced Z-Score Mean Reversion Strategy (trend-aware)

Features:
- Z-score entry/mean-reversion exit
- RSI + BB width + (optional) volume filters
- ATR stop/target + trailing stop
- Max holding period + cooldown after losses
- Slippage + transaction fees (added)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from state import AppState
from ui.statusbar import add_text_status, add_text_status_backtest


# -----------------------------
# Indicator helpers
# -----------------------------
def calc_rolling_mean(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def calc_rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=window).std()


def calc_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    mean = calc_rolling_mean(series, window=window)
    std = calc_rolling_std(series, window=window).replace(0, np.nan)
    return (series - mean) / std


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


def calc_bollinger_band_width(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    sma = calc_rolling_mean(series, window)
    std = calc_rolling_std(series, window)
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return ((upper - lower) / sma) * 100


def calc_volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume.rolling(window=window, min_periods=window).mean()


# -----------------------------
# Execution cost helpers (added)
# -----------------------------
def apply_slippage(price: float, position: int, is_entry: bool, slippage_bps: float) -> float:
    """
    position:  1 (long) or -1 (short)
    is_entry: True for entry fills, False for exit fills
    slippage_bps: e.g. 5.0 = 5 bps = 0.05%
    """
    slip = slippage_bps / 10000.0
    if position == 1:
        # Long: entry worse (higher), exit worse (lower)
        return price * (1.0 + slip) if is_entry else price * (1.0 - slip)
    else:
        # Short: entry worse (lower), exit worse (higher)
        return price * (1.0 - slip) if is_entry else price * (1.0 + slip)


def calc_fee(notional: float, fee_bps: float) -> float:
    """
    fee_bps: per-side fee in basis points, applied to notional each entry/exit
    """
    return notional * (fee_bps / 10000.0)


# -----------------------------
# Strategy
# -----------------------------
def zscore_strategy_rw(state: AppState) -> pd.DataFrame:
    add_text_status(state, "Z-Score Strategy: Starting...")

    # ---- Load & validate data ----
    try:
        if not hasattr(state, "csv_data") or state.csv_data is None:
            add_text_status(state, "Z-Score: ERROR - No CSV data loaded.")
            state.backtest_results = pd.DataFrame()
            return state.backtest_results

        df = state.csv_data.copy()
        add_text_status(state, f"Z-Score: Loaded {len(df)} rows of data.")
    except Exception as e:
        add_text_status(state, f"Z-Score: ERROR reading data - {e}")
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    if "Close" not in df.columns:
        add_text_status(state, "Z-Score: ERROR - Missing required 'Close' column.")
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Create synthetic OHLC if missing
    if "Open" not in df.columns:
        df["Open"] = df["Close"].shift(1).fillna(df["Close"])
        add_text_status(state, "Z-Score: Created synthetic 'Open' from Close.")
    else:
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")

    if "High" not in df.columns:
        df["High"] = df["Close"]
        add_text_status(state, "Z-Score: Created synthetic 'High' from Close.")
    else:
        df["High"] = pd.to_numeric(df["High"], errors="coerce")

    if "Low" not in df.columns:
        df["Low"] = df["Close"]
        add_text_status(state, "Z-Score: Created synthetic 'Low' from Close.")
    else:
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

    has_volume = "Volume" in df.columns
    if has_volume:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    length = len(df)
    min_required_bars = 120
    if length < min_required_bars:
        add_text_status(state, f"Z-Score: ERROR - Need at least {min_required_bars} bars, got {length}.")
        state.backtest_results = pd.DataFrame()
        return state.backtest_results

    add_text_status(state, f"Z-Score: Data validation passed. {length} bars available.")

    # ---- Indicators ----
    add_text_status(state, "Z-Score: Calculating indicators...")

    df["z_score"] = calc_z_score(df["Close"], window=20)
    df["rsi"] = calc_rsi(df["Close"], window=14)
    df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], window=14)

    df["ma_100"] = calc_rolling_mean(df["Close"], window=100)
    df["ma_50"] = calc_rolling_mean(df["Close"], window=50)

    denom = df["ma_50"].shift(20).replace(0, np.nan)
    df["ma_50_slope"] = ((df["ma_50"] - df["ma_50"].shift(20)) / denom) * 100

    df["bb_width"] = calc_bollinger_band_width(df["Close"], window=20, num_std=2.0)

    if has_volume:
        df["volume_sma"] = calc_volume_sma(df["Volume"], window=20)

    add_text_status(state, "Z-Score: Indicators calculated successfully.")

    # ---- Parameters ----
    uptrend_slope_threshold = 0.5
    downtrend_slope_threshold = -0.5

    z_entry_long_trend = -1.5
    z_entry_long_counter = -2.5
    z_entry_short_trend = 1.5
    z_entry_short_counter = 2.5

    z_exit_long_trend = 1.0
    z_exit_long_counter = 0.0
    z_exit_short_trend = -1.0
    z_exit_short_counter = 0.0

    rsi_long_min, rsi_long_max = 20, 60
    rsi_short_min, rsi_short_max = 40, 80

    atr_stop_multiplier = 2.0
    atr_target_multiplier = 4.0
    atr_trail_activation = 1.5
    atr_trail_distance = 1.0

    max_holding_bars = 80
    cooldown_bars = 3

    bb_width_valid = df["bb_width"].dropna()
    bb_width_threshold = bb_width_valid.quantile(0.20) if not bb_width_valid.empty else 0.0

    starting_balance = 100000.0
    trade_size = 2000.0  # Edited
    leverage = 10.0

    # ---- Slippage + fees (added) ----
    slippage_bps = 5.0          # e.g. 5 bps = 0.05% per fill
    fee_bps = 4.0               # e.g. 4 bps per side (entry and exit)
    notional = trade_size * leverage

    add_text_status(state, f"Z-Score: Params set. BB width threshold: {bb_width_threshold:.2f}%")

    # ---- Trading state ----
    account_balance = starting_balance

    position = 0  # 0 flat, 1 long, -1 short
    entry_price: Optional[float] = None
    entry_idx: Optional[int] = None
    entry_atr: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_active = False
    highest_since_entry: Optional[float] = None
    lowest_since_entry: Optional[float] = None

    bars_since_last_trade = 999
    last_trade_was_loss = False

    trade_dates = []
    cumulative_pct_returns = []
    account_balance_history = []
    trade_sides = []
    trade_entry_prices = []
    trade_exit_prices = []
    trade_pnl = []
    trade_exit_reasons = []

    add_text_status(state, "Z-Score: Beginning bar-by-bar scan...")

    # ---- Main loop ----
    for i in range(length):
        current_close = df.at[i, "Close"]
        current_high = df.at[i, "High"]
        current_low = df.at[i, "Low"]
        current_z = df.at[i, "z_score"]
        current_rsi = df.at[i, "rsi"]
        current_atr = df.at[i, "atr"]
        current_ma100 = df.at[i, "ma_100"]
        current_ma50 = df.at[i, "ma_50"]
        current_ma_slope = df.at[i, "ma_50_slope"]
        current_bb_width = df.at[i, "bb_width"]

        current_volume = df.at[i, "Volume"] if has_volume else np.nan
        current_volume_sma = df.at[i, "volume_sma"] if has_volume else np.nan

        # Indicators ready?
        if (
            pd.isna(current_close)
            or pd.isna(current_z)
            or pd.isna(current_rsi)
            or pd.isna(current_atr)
            or pd.isna(current_ma100)
            or pd.isna(current_ma50)
            or pd.isna(current_ma_slope)
        ):
            bars_since_last_trade += 1
            continue

        bars_since_last_trade += 1

        is_uptrend = current_ma_slope > uptrend_slope_threshold
        is_downtrend = current_ma_slope < downtrend_slope_threshold

        # =========================
        # Exit logic (if in position)
        # =========================
        if position != 0:
            exit_triggered = False
            exit_reason = ""
            exit_price = float(current_close)

            # Track highs/lows for trailing
            if position == 1:
                if highest_since_entry is None or current_high > highest_since_entry:
                    highest_since_entry = float(current_high)
            else:
                if lowest_since_entry is None or current_low < lowest_since_entry:
                    lowest_since_entry = float(current_low)

            # 1) Mean reversion exit
            if position == 1:
                z_exit_threshold = z_exit_long_trend if is_uptrend else z_exit_long_counter
                if current_z >= z_exit_threshold:
                    exit_triggered = True
                    exit_reason = "Mean Reversion"
            else:
                z_exit_threshold = z_exit_short_trend if is_downtrend else z_exit_short_counter
                if current_z <= z_exit_threshold:
                    exit_triggered = True
                    exit_reason = "Mean Reversion"

            # 2) Stop loss
            if not exit_triggered and stop_loss is not None:
                if position == 1 and current_low <= stop_loss:
                    exit_triggered = True
                    exit_price = float(stop_loss)
                    exit_reason = "Stop Loss"
                elif position == -1 and current_high >= stop_loss:
                    exit_triggered = True
                    exit_price = float(stop_loss)
                    exit_reason = "Stop Loss"

            # 3) Take profit
            if not exit_triggered and take_profit is not None:
                if position == 1 and current_high >= take_profit:
                    exit_triggered = True
                    exit_price = float(take_profit)
                    exit_reason = "Take Profit"
                elif position == -1 and current_low <= take_profit:
                    exit_triggered = True
                    exit_price = float(take_profit)
                    exit_reason = "Take Profit"

            # 4) Trailing stop
            if not exit_triggered and trailing_active and entry_atr is not None:
                if position == 1 and highest_since_entry is not None:
                    trail_stop = highest_since_entry - (entry_atr * atr_trail_distance)
                    if current_low <= trail_stop:
                        exit_triggered = True
                        exit_price = float(trail_stop)
                        exit_reason = "Trailing Stop"
                elif position == -1 and lowest_since_entry is not None:
                    trail_stop = lowest_since_entry + (entry_atr * atr_trail_distance)
                    if current_high >= trail_stop:
                        exit_triggered = True
                        exit_price = float(trail_stop)
                        exit_reason = "Trailing Stop"

            # Activate trailing stop if enough profit
            if (
                not exit_triggered
                and not trailing_active
                and entry_atr is not None
                and entry_price is not None
            ):
                profit_distance = entry_atr * atr_trail_activation
                if position == 1 and (current_close - entry_price) >= profit_distance:
                    trailing_active = True
                elif position == -1 and (entry_price - current_close) >= profit_distance:
                    trailing_active = True

            # 5) Time exit
            if not exit_triggered and entry_idx is not None:
                if (i - entry_idx) >= max_holding_bars:
                    exit_triggered = True
                    exit_reason = "Time Exit"

            # Execute exit
            if exit_triggered and entry_price is not None:
                # Apply exit slippage (added)
                exit_price = float(apply_slippage(float(exit_price), position=position, is_entry=False, slippage_bps=slippage_bps))

                if position == 1:
                    pct_move = (exit_price - entry_price) / entry_price * 100.0
                else:
                    pct_move = (entry_price - exit_price) / entry_price * 100.0

                gross_pnl = (trade_size * leverage) * (pct_move / 100.0)

                # Apply exit fee (added)
                exit_fee = calc_fee(notional, fee_bps)
                pnl = gross_pnl - exit_fee
                account_balance += pnl

                last_trade_was_loss = pnl < 0
                bars_since_last_trade = 0

                trade_date = df.at[i, "Date"] if "Date" in df.columns else None
                trade_dates.append(trade_date)
                cumulative_pct_returns.append((account_balance - starting_balance) / starting_balance * 100.0)
                account_balance_history.append(account_balance)
                trade_sides.append("Long" if position == 1 else "Short")
                trade_entry_prices.append(entry_price)
                trade_exit_prices.append(exit_price)
                trade_pnl.append(pnl)
                trade_exit_reasons.append(exit_reason)

                side_str = "Long" if position == 1 else "Short"
                add_text_status_backtest(
                    state,
                    f"CLOSED {side_str} @ {exit_price:.2f} | Reason: {exit_reason} | "
                    f"PnL: ${pnl:+.2f} | Balance: ${account_balance:.2f}"
                )

                # Reset position state
                position = 0
                entry_price = None
                entry_idx = None
                entry_atr = None
                stop_loss = None
                take_profit = None
                trailing_active = False
                highest_since_entry = None
                lowest_since_entry = None

            continue

        # =========================
        # Entry logic (flat)
        # =========================
        if last_trade_was_loss and bars_since_last_trade < cooldown_bars:
            continue

        if account_balance < trade_size:
            add_text_status_backtest(state, "Z-Score: Insufficient balance for new trade.")
            break

        if pd.isna(current_bb_width) or current_bb_width < bb_width_threshold:
            continue

        if has_volume and not pd.isna(current_volume_sma):
            if pd.isna(current_volume) or current_volume < current_volume_sma * 0.8:
                continue

        z_long_threshold = z_entry_long_trend if is_uptrend else z_entry_long_counter
        z_short_threshold = z_entry_short_trend if is_downtrend else z_entry_short_counter

        long_signal = (
            (current_z < z_long_threshold)
            and (rsi_long_min <= current_rsi <= rsi_long_max)
            and (current_close >= current_ma50 * 0.92)
        )

        short_signal = False
        if not is_uptrend:
            short_signal = (
                (current_z > z_short_threshold)
                and (rsi_short_min <= current_rsi <= rsi_short_max)
                and (current_close <= current_ma50 * 1.08)
            )

        if long_signal:
            position = 1

            # Apply entry slippage (added)
            raw_entry = float(current_close)
            entry_price = float(apply_slippage(raw_entry, position=position, is_entry=True, slippage_bps=slippage_bps))

            # Apply entry fee (added)
            entry_fee = calc_fee(notional, fee_bps)
            account_balance -= entry_fee

            entry_idx = i
            entry_atr = float(current_atr)

            stop_mult = atr_stop_multiplier
            target_mult = atr_target_multiplier if is_uptrend else atr_target_multiplier * 0.75

            stop_loss = entry_price - (entry_atr * stop_mult)
            take_profit = entry_price + (entry_atr * target_mult)

            trailing_active = False
            highest_since_entry = float(current_high)
            lowest_since_entry = float(current_low)

            trend_str = "UPTREND" if is_uptrend else ("DOWNTREND" if is_downtrend else "NEUTRAL")
            add_text_status_backtest(
                state,
                f"OPENED LONG @ {entry_price:.2f} | Z: {current_z:.2f} | {trend_str} | "
                f"SL: {stop_loss:.2f} | TP: {take_profit:.2f}"
            )

        elif short_signal:
            position = -1

            # Apply entry slippage (added)
            raw_entry = float(current_close)
            entry_price = float(apply_slippage(raw_entry, position=position, is_entry=True, slippage_bps=slippage_bps))

            # Apply entry fee (added)
            entry_fee = calc_fee(notional, fee_bps)
            account_balance -= entry_fee

            entry_idx = i
            entry_atr = float(current_atr)

            stop_mult = atr_stop_multiplier
            target_mult = atr_target_multiplier if is_downtrend else atr_target_multiplier * 0.75

            stop_loss = entry_price + (entry_atr * stop_mult)
            take_profit = entry_price - (entry_atr * target_mult)

            trailing_active = False
            highest_since_entry = float(current_high)
            lowest_since_entry = float(current_low)

            add_text_status_backtest(
                state,
                f"OPENED SHORT @ {entry_price:.2f} | Z: {current_z:.2f} | RSI: {current_rsi:.1f} | "
                f"SL: {stop_loss:.2f} | TP: {take_profit:.2f}"
            )

    # ---- Close any open position at end ----
    if position != 0 and entry_price is not None:
        exit_idx = length - 1
        exit_price = float(df.at[exit_idx, "Close"])

        # Apply exit slippage (added)
        exit_price = float(apply_slippage(float(exit_price), position=position, is_entry=False, slippage_bps=slippage_bps))

        if position == 1:
            pct_move = (exit_price - entry_price) / entry_price * 100.0
        else:
            pct_move = (entry_price - exit_price) / entry_price * 100.0

        gross_pnl = (trade_size * leverage) * (pct_move / 100.0)

        # Apply exit fee (added)
        exit_fee = calc_fee(notional, fee_bps)
        pnl = gross_pnl - exit_fee
        account_balance += pnl

        trade_date = df.at[exit_idx, "Date"] if "Date" in df.columns else None
        trade_dates.append(trade_date)
        cumulative_pct_returns.append((account_balance - starting_balance) / starting_balance * 100.0)
        account_balance_history.append(account_balance)
        trade_sides.append("Long" if position == 1 else "Short")
        trade_entry_prices.append(entry_price)
        trade_exit_prices.append(exit_price)
        trade_pnl.append(pnl)
        trade_exit_reasons.append("End of Data")

        side_str = "Long" if position == 1 else "Short"
        add_text_status_backtest(
            state,
            f"CLOSED {side_str} @ EOD {exit_price:.2f} | PnL: ${pnl:+.2f} | Balance: ${account_balance:.2f}"
        )

    # ---- Results ----
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

    total_trades = len(trade_pnl)
    winning_trades = sum(1 for p in trade_pnl if p > 0)
    losing_trades = sum(1 for p in trade_pnl if p < 0)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = float(np.nansum(trade_pnl)) if trade_pnl else 0.0

    add_text_status_backtest(state, "=" * 50)
    add_text_status_backtest(state, "Z-SCORE STRATEGY SUMMARY")
    add_text_status_backtest(state, "=" * 50)
    add_text_status_backtest(state, f"Total Trades: {total_trades}")
    add_text_status_backtest(state, f"Winning Trades: {winning_trades}")
    add_text_status_backtest(state, f"Losing Trades: {losing_trades}")
    add_text_status_backtest(state, f"Win Rate: {win_rate:.1f}%")
    add_text_status_backtest(state, f"Total PnL: ${total_pnl:+.2f}")
    add_text_status_backtest(state, f"Final Balance: ${account_balance:.2f}")
    add_text_status_backtest(
        state,
        f"Return: {(account_balance - starting_balance) / starting_balance * 100:+.2f}%",
    )
    add_text_status_backtest(state, "=" * 50)

    time.sleep(0.2)
    return final_results_df


# Export names for strategy registration
strategy_name = "zscore_strategy_rw"
strategy_entry = zscore_strategy_rw

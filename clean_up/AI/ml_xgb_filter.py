"""
AI/ML — XGBoost Trade-Quality Filter (GPU)  ✅ robust early-stopping

What this does:
- Builds candidate entries from your existing "best" signal logic (TREND + RANGE)
- Labels each candidate with a fixed-horizon barrier outcome:
    label=1 if TP hit before SL within horizon, else 0
- Trains a GPU XGBoost Booster via xgb.train() (works across xgboost versions)
- Saves:
    models/eth_xgb_filter.json
    models/eth_xgb_filter_features.json
- Stores into state:
    state.ml_booster
    state.ml_feature_names
    state.ml_last_train_summary

Drop this file at:
  clean_up/AI/ml_xgb_filter.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


# =============================
# Indicator helpers
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

    plus_di = 100 * (
        pd.Series(plus_dm, index=close.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=close.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        / atr.replace(0, np.nan)
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx


# =============================
# 4h regime from 15m
# =============================
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
# Candidate generator (matches best strategy)
# =============================
@dataclass(frozen=True)
class BaseSignalParams:
    adx_trend_threshold: float = 20.0
    slope_trend_threshold: float = 0.1
    trend_z_entry: float = 1.0
    range_z_extreme: float = 2.4
    trend_rsi_reclaim: float = 46.0


def generate_candidate_entries(df: pd.DataFrame, p: BaseSignalParams) -> pd.DataFrame:
    adx4 = df["adx"].values
    slope4 = df["ema_50_slope_pct"].values
    resid_z = df["resid_z"].values
    rsi = df["rsi"].values

    trend_regime = (adx4 >= p.adx_trend_threshold) & (np.abs(slope4) >= p.slope_trend_threshold)
    regime = np.where(trend_regime, "TREND", "RANGE")
    trend_dir = np.where(slope4 > 0, 1, np.where(slope4 < 0, -1, 0))

    long_sig = np.zeros(len(df), dtype=bool)
    short_sig = np.zeros(len(df), dtype=bool)

    uptrend = (regime == "TREND") & (trend_dir == 1)
    downtrend = (regime == "TREND") & (trend_dir == -1)

    # TREND pullback
    long_sig |= uptrend & (resid_z <= -p.trend_z_entry) & (rsi >= p.trend_rsi_reclaim)
    short_sig |= downtrend & (resid_z >= p.trend_z_entry) & (rsi <= (100.0 - p.trend_rsi_reclaim))

    # RANGE mean reversion
    range_reg = (regime == "RANGE")
    long_sig |= range_reg & (resid_z <= -p.range_z_extreme) & (rsi <= 35.0)
    short_sig |= range_reg & (resid_z >= p.range_z_extreme) & (rsi >= 65.0)

    rows = []
    for i in range(len(df)):
        if long_sig[i]:
            rows.append((i, 1, regime[i]))
        elif short_sig[i]:
            rows.append((i, -1, regime[i]))

    return pd.DataFrame(rows, columns=["idx", "side", "regime"])


# =============================
# Fixed-horizon barrier labels
# =============================
def _first_true_index(mask: np.ndarray) -> Optional[int]:
    if not mask.any():
        return None
    return int(np.argmax(mask))


def label_candidates_fixed_horizon(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    horizon_bars: int = 48,
    target_pct: float = 0.0064,
    stop_pct: float = 0.0048,
) -> np.ndarray:
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    y = np.zeros(len(candidates), dtype=np.int32)
    n = len(df)

    for j, row in enumerate(candidates.itertuples(index=False)):
        i = int(row.idx)
        side = int(row.side)
        entry = float(close[i])

        end = min(n - 1, i + horizon_bars)
        if end <= i + 1:
            y[j] = 0
            continue

        h = high[i + 1 : end + 1]
        l = low[i + 1 : end + 1]

        if side == 1:
            tp_level = entry * (1.0 + target_pct)
            sl_level = entry * (1.0 - stop_pct)
            tp_hit = h >= tp_level
            sl_hit = l <= sl_level
        else:
            tp_level = entry * (1.0 - target_pct)
            sl_level = entry * (1.0 + stop_pct)
            tp_hit = l <= tp_level
            sl_hit = h >= sl_level

        tp_i = _first_true_index(tp_hit)
        sl_i = _first_true_index(sl_hit)

        if tp_i is None and sl_i is None:
            y[j] = 0
        elif tp_i is None:
            y[j] = 0
        elif sl_i is None:
            y[j] = 1
        else:
            y[j] = 1 if tp_i < sl_i else 0

    return y


# =============================
# Feature engineering (entry-time only)
# =============================
def build_feature_frame(df: pd.DataFrame, candidates: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    idx = candidates["idx"].values.astype(int)
    feat = pd.DataFrame(index=candidates.index)

    # Core strategy features
    feat["resid_z"] = df.loc[idx, "resid_z"].values
    feat["rsi"] = df.loc[idx, "rsi"].values
    feat["atrp"] = df.loc[idx, "atrp"].values
    feat["adx4"] = df.loc[idx, "adx"].values
    feat["slope4"] = df.loc[idx, "ema_50_slope_pct"].values

    # Price context
    feat["close"] = df.loc[idx, "Close"].values
    feat["fair"] = df.loc[idx, "fair"].values
    feat["dist_fair_pct"] = (feat["close"] - feat["fair"]) / feat["fair"].replace(0, np.nan)

    # Candle shape
    o = df.loc[idx, "Open"].values
    h = df.loc[idx, "High"].values
    l = df.loc[idx, "Low"].values
    c = df.loc[idx, "Close"].values
    rng = np.maximum(h - l, 1e-9)

    feat["candle_body_pct"] = (c - o) / np.maximum(o, 1e-9)
    feat["candle_range_pct"] = rng / np.maximum(o, 1e-9)
    feat["upper_wick_frac"] = (h - np.maximum(o, c)) / rng
    feat["lower_wick_frac"] = (np.minimum(o, c) - l) / rng

    # Recent returns (momentum)
    close = df["Close"]
    feat["ret_1"] = close.pct_change(1).iloc[idx].values
    feat["ret_4"] = close.pct_change(4).iloc[idx].values
    feat["ret_12"] = close.pct_change(12).iloc[idx].values
    feat["ret_48"] = close.pct_change(48).iloc[idx].values

    # Encodings
    feat["side"] = candidates["side"].values.astype(int)
    feat["is_trend_regime"] = (candidates["regime"].values == "TREND").astype(int)
    feat["trend_dir"] = np.where(feat["slope4"] > 0, 1, np.where(feat["slope4"] < 0, -1, 0))

    # Time features
    dt = pd.to_datetime(df.loc[idx, "Date"].values)
    feat["hour"] = dt.hour.astype(int)
    feat["dow"] = dt.dayofweek.astype(int)

    # Clean
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat, feat.columns.tolist()


# =============================
# Public: Train + Save (GUI callback)
# =============================
def train_xgb_filter_model(
    state,
    horizon_bars: int = 48,
    target_pct: float = 0.0064,
    stop_pct: float = 0.0048,
    train_split: float = 0.70,
) -> Dict[str, Any]:
    if not hasattr(state, "csv_data") or state.csv_data is None:
        raise RuntimeError("No CSV data loaded in state.csv_data")

    df = state.csv_data.copy()

    # --- Validate and coerce ---
    if "Date" not in df.columns:
        raise RuntimeError("Data must include a 'Date' column")
    if "Close" not in df.columns:
        raise RuntimeError("Data must include a 'Close' column")

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

    # --- Indicators (same as strategy) ---
    df["fair"] = calc_ema(df["Close"], span=50)
    df["residual"] = df["Close"] - df["fair"]
    resid_std = df["residual"].rolling(window=80, min_periods=80).std().replace(0, np.nan)
    df["resid_z"] = df["residual"] / resid_std

    df["rsi"] = calc_rsi(df["Close"], window=14)
    df["atr"] = calc_atr(df["High"], df["Low"], df["Close"], window=14)
    df["atrp"] = df["atr"] / df["Close"].replace(0, np.nan)

    df4 = build_4h_regime_from_15m(df)
    df = attach_4h_regime_to_15m(df, df4)

    df = df.dropna(
        subset=["Date", "Open", "High", "Low", "Close", "rsi", "atrp", "resid_z", "adx", "ema_50_slope_pct"]
    ).reset_index(drop=True)

    if len(df) < 5000:
        raise RuntimeError(f"Not enough clean rows after indicators: {len(df)}")

    # --- Candidates and labels ---
    candidates = generate_candidate_entries(df, BaseSignalParams())
    if candidates.empty:
        raise RuntimeError("No candidate entries generated from signal rules")

    y = label_candidates_fixed_horizon(
        df, candidates, horizon_bars=horizon_bars, target_pct=target_pct, stop_pct=stop_pct
    )

    # --- Features ---
    X, feature_names = build_feature_frame(df, candidates)
    y_aligned = y[X.index.values.astype(int)]

    # --- Time split (70/30) ---
    n = len(X)
    split = int(n * train_split)
    X_train, y_train = X.iloc[:split], y_aligned[:split]
    X_test, y_test = X.iloc[split:], y_aligned[split:]

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # --- GPU training (robust across versions) ---
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",

        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "lambda": 1.0,
        "alpha": 0.0,
        "gamma": 0.0,
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=8000,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=50,
    )

    # --- quick eval ---
    p_test = booster.predict(dtest)
    pred_test = (p_test >= 0.50).astype(int)
    acc = float((pred_test == y_test).mean())
    pos_rate = float(np.mean(y_test))

    # --- save ---
    model_path = getattr(state, "ml_model_path", "models/eth_xgb_filter.json")
    feat_path = getattr(state, "ml_feature_path", "models/eth_xgb_filter_features.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    booster.save_model(model_path)
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    summary = {
        "candidates": int(len(candidates)),
        "samples_used": int(n),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_accuracy@0.5": acc,
        "test_positive_rate": pos_rate,
        "best_iteration": int(booster.best_iteration or 0),
        "horizon_bars": int(horizon_bars),
        "target_pct": float(target_pct),
        "stop_pct": float(stop_pct),
        "model_path": model_path,
        "feature_path": feat_path,
    }

    # store in state
    state.ml_last_train_summary = summary
    state.ml_booster = booster
    state.ml_feature_names = feature_names

    print("[ML TRAIN] Saved:", model_path)
    print("[ML TRAIN] Test acc@0.5:", acc, "| pos_rate:", pos_rate, "| best_iter:", summary["best_iteration"])
    return summary


def load_xgb_filter_model(state) -> Tuple[xgb.Booster, List[str]]:
    model_path = getattr(state, "ml_model_path", "models/eth_xgb_filter.json")
    feat_path = getattr(state, "ml_feature_path", "models/eth_xgb_filter_features.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature list not found: {feat_path}")

    with open(feat_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    booster = xgb.Booster()
    booster.load_model(model_path)
    print("[ML] Booster attributes:", booster.attributes())
    state.ml_booster = booster
    state.ml_feature_names = feature_names
    return booster, feature_names

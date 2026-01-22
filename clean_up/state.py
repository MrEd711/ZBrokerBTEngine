# state.py
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from typing import Optional

@dataclass
class AppState:
    csv_path: Optional[Path] = None
    csv_data: Optional[pd.DataFrame] = None
    sp500_data: Optional[pd.DataFrame] = None
    selected_interval: str = ""
    status_height: int = 200
    ui_ready: bool = False
    selected_asset: str = ""
    sp500_checkbox: bool = True

    


    # Additions for backtesting DATA

    backtest_csv: Optional[Path] = None
    backtest_results: Optional[pd.DataFrame] = None
    backtest_results_list: list = field(default_factory=list)

    buy_hold: Optional[pd.DataFrame] = None
    

    # Metrics for backtesting
    # Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, CAGR, MAE

    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    cagr: Optional[float] = None
    mae: Optional[float] = None
    total_trades: Optional[int] = None
    # Possibly more (Needs to be integrated into strategy logic)
    metrics_df: Optional[pd.DataFrame] = None



    # Indicators

    ema_data_values: Optional[pd.Series] = None
    ema_period: int = 200

    sma_fast: int = 10
    sma_slow: int = 50


    # MACD
    ena_close_fast: int = 12
    ena_close_slow: int = 26
    ena_close_signal: int = 9
    macd_values: Optional[pd.DataFrame] = None

    # RSI
    rsi_period: int = 14
    # Needs a smoothing (Wilder is standard)
    rsi_values: Optional[pd.DataFrame] = None
    # Sorce: close is standard

    # ROC

    roc_period: int = 12
    roc_values: Optional[pd.DataFrame] = None

    

    ### AI
    ml_enabled: bool = True
    ml_model_path: str = "models/eth_xgb_filter.json"
    ml_feature_path: str = "models/eth_xgb_filter_features.json"

    ml_threshold: float = 0.60          # hard gate
    ml_min_scale: float = 0.35          # size scaling floor for p==threshold
    ml_max_scale: float = 1.00          # size scaling cap
    ml_last_train_summary: dict = None  # store metrics from last training

    ml_horizon_bars: int = 48
    ml_target_pct: float = 0.0064
    ml_stop_pct: float = 0.0048



    # Chart Mode
    volume_chart = False

    # Other settings backtest window
    show_chart_main: bool = False
    save_csv_backtest: bool = False
    
    

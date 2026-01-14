# actions/backtest.py
import sys, json, subprocess, pathlib, pandas as pd, dearpygui.dearpygui as dpg
from state import AppState
from ui.statusbar import add_text_status
from STRATEGIES.strategy_pt import simple_strategy, confluence_based_strategy, sma_crossover_strategy
from ui.charts import generate_chart
from STRATEGIES.zbroker_strat import zscore_strategy
from STRATEGIES.calculate_metrics import metrics_df, compute_and_assign_all

def backtest_strategy(state: AppState, strategy_name: str):

    if state.csv_data is None or state.csv_path is None:
        add_text_status(state, "No CSV loaded for backtesting.")
        return
    if strategy_name == "Simple Strategy":
        simple_strategy(state)
    elif strategy_name == "Confluence Based Strategy":
        confluence_based_strategy(state)
    elif strategy_name == "SMA Crossover":
        sma_crossover_strategy(state)
    elif strategy_name == "Please Select":
        add_text_status(state, "Please select a valid strategy.")
        return
    elif strategy_name == "ZBroker":
        zscore_strategy(state)
    else:
        add_text_status(state, "Error")
        return
    

    if state.save_csv_backtest:
        df = state.backtest_results
        df.to_csv(f"clean_up/backtest_saves/backtest_results_{strategy_name}.csv", index=False) # Workaround
        
        # If main chart is not shown then show the main chart
    if not dpg.is_item_shown("chart") and state.show_chart_main == True:
        generate_chart(state)

    compute_and_assign_all(state)
    dpg.set_value("metrics_text", str(state.metrics_df))
    add_text_status(state, f"Backtest complete using {strategy_name}. Metrics updated.")
    dpg.show_item("metrics_backtest")

    equity_plot(state)

def equity_plot(state: AppState):
    try:
        df = pd.read_csv("sp500.csv")
    except Exception as e:
        add_text_status(state, f"No S&P 500 data: {e}")
        return

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Return_%"])

    # Remove any previous series
    if dpg.does_item_exist("equity_series"):
        dpg.delete_item("equity_series")
    if dpg.does_item_exist("backtest_equity_series"):
        dpg.delete_item("backtest_equity_series")
    if dpg.does_item_exist("buy_hold_equity_series"):
        dpg.delete_item("buy_hold_equity_series")

    # --- S&P 500 line: ensure float lists ---
    x = ((df["Date"].astype("int64") / 1e9).astype(float)).tolist()
    y = (df["Return_%"].astype(float)).tolist()
    dpg.show_item("equity_plot")
    dpg.add_line_series(
        x, y,
        parent="y_axis_equity",
        tag="equity_series",
        label="S&P 500 %", 
        skip_nan=True,
    )




    bh_x = ((state.buy_hold["Date"].astype("int64") / 1e9).astype(float)).tolist()
    bh_y = (state.buy_hold["Cumulative Percentage Returns"].astype(float)).tolist()

    dpg.add_line_series(
        bh_x, bh_y,
        parent="y_axis_equity",
        tag="buy_hold_equity_series",
        label="Buy and Hold Equity",
        skip_nan=True
    )


    # --- Backtest line: ensure float lists ---
    if state.backtest_results is not None and not state.backtest_results.empty:
        br = state.backtest_results.dropna(subset=["Date", "Cumulative Percentage Returns"]).copy()
        backtest_x = ((br["Date"].astype("int64") / 1e9).astype(float)).tolist()
        backtest_y = (br["Cumulative Percentage Returns"].astype(float)).tolist()

        # lengths must match
        n = min(len(backtest_x), len(backtest_y))
        backtest_x, backtest_y = backtest_x[:n], backtest_y[:n]

        dpg.add_line_series(
            backtest_x, backtest_y,
            parent="y_axis_equity",
            tag="backtest_equity_series",
            label=f'{dpg.get_value("strategy_combo")} Equity',
            skip_nan=True        
        )

def reload_equity_plot(state: AppState):
    if not dpg.is_item_shown("equity_plot"):
        if dpg.does_item_exist("equity_series") or dpg.does_item_exist("backtest_equity_series"):
            #dpg.delete_item("equity_series")
            #equity_plot(state) FIX THIS because lines duplicate on reload causing errors. Could skip past pre-loaded lines and only calculate new ones.
            dpg.show_item("equity_plot")
        else:
            add_text_status(state, "No equity curve loaded")





    
        
    



    

# actions/dataflow.py
import sys, subprocess, pathlib, pandas as pd, dearpygui.dearpygui as dpg
from state import AppState
from ui.statusbar import add_text_status
from actions.buy_hold_script import buy_and_hold_strategy

INTERVAL_MAP = {"1H":"1h","2H":"2h","4H":"4h"}

def set_ema_values(state: AppState, df):
    if len(df) > state.ema_period + 3:
            state.ema_data_values = df["Close"].ewm(span=state.ema_period, adjust=False).mean() # EMA ADDED

def on_load_csv(state: AppState, sender, app_data):
    path = pathlib.Path(app_data["file_path_name"])
    if path.suffix.lower() != ".csv":
        add_text_status(state, "Please select a CSV file.")
        return
    try:
        df = pd.read_csv(path)
        state.csv_path = path
        state.csv_data = df
        set_ema_values(state, df)  # Set EMA values when loading CSV
        add_text_status(state, f"CSV loaded: {path.name}")
        dpg.set_value("CSV_CURRENT", f"Current CSV: {str(path.name)}")
    except Exception as e:
        add_text_status(state, f"Error loading CSV: {e}")

    try:
        buy_and_hold_strategy(state)
    except Exception as e:
        add_text_status(state, f"Error calculating Buy and Hold: {e}")


def fetch_eth_via_binance(state: AppState, days: int, out: str, script_path: pathlib.Path, interval_ui: str):
    if dpg.is_item_shown("chart"):
        dpg.configure_item("chart", show=False)
    for tag in ("candles",):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    interval = INTERVAL_MAP.get(interval_ui, interval_ui)  # normalise
    cmd = [
    sys.executable,
    str(script_path),
    "--days", str(days),
    "--out", out,
    "--interval", interval,
    "--symbol", state.selected_asset,   # <-- added
]

    # Pass S&P 500 fetch preference to the script so subprocess honors UI checkbox
    if not state.sp500_checkbox:
        cmd.append("--no-sp500")

    add_text_status(state, f"Running: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)  # will raise on error

    csv_file = pathlib.Path(out)
    if not csv_file.is_file():
        add_text_status(state, f"Download OK but CSV missing: {out}")
        return

    try:
        df = pd.read_csv(csv_file)
        state.csv_path = csv_file            # ← important: keep path in sync
        state.csv_data = df
        add_text_status(state, f"Auto-loaded CSV ({len(df)} rows) → {out}")
        dpg.set_value("CSV_CURRENT", f"Current CSV: {str(csv_file)}")
        if dpg.is_item_shown("data_entry"):
            dpg.configure_item("data_entry", show=False)
    except Exception as e:
        add_text_status(state, f"Error auto-loading CSV: {e}")

def file_dialog_download_cb(state: AppState, sender, app_data):
    state.selected_asset = str(dpg.get_value("asset_combo"))
    state.selected_interval = str(dpg.get_value("interval_combo"))
    try:
        days = int(dpg.get_value("days_input"))
    except Exception as e:
        dpg.set_value("error_text", f"Days must be an integer. Error: {e}")
        return
    if not (1 <= days <= 1500):
        dpg.set_value("error_text", "Selected days must be in range 1 → 365")
        return
    add_text_status(state, "Downloading via fetch_eth_csv.py …")
    fetch_eth_via_binance(
        state=state,
        days=days,
        out=f"{state.selected_asset}_{state.selected_interval}.csv",
        script_path=pathlib.Path(app_data["file_path_name"]),
        interval_ui=state.selected_interval
    )
    # Edit to add POLUSDT

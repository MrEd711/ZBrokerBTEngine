# ui/charts.py
import time
import pandas as pd
import dearpygui.dearpygui as dpg
from typing import Optional
from state import AppState
from ui.statusbar import add_text_status
from actions.dataflow import set_ema_values

def ensure_time_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    return df

def generate_chart(state: AppState):
    if dpg.does_item_exist("candles"):
        dpg.delete_item("candles")
    
    if state.csv_data is None:
        return
    df = ensure_time_col(state.csv_data)

    o = df["Open"].tolist()
    h = df["High"].tolist()
    l = df["Low"].tolist()
    c = df["Close"].tolist()
    #x = (df["Date"].astype("int64") // 10**9).tolist()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    x = (pd.to_datetime(df["Date"]).values.astype("int64") // 10**9).tolist()

    v = df["Volume"].tolist()

    for tag in ("candles", "volume_stem"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    dpg.show_item("chart")
    
    if not state.volume_chart:
        dpg.add_candle_series(x, o, c, l, h,
            parent="y_axis",
            tag="candles",
            weight=0.5,
            time_unit=dpg.mvTimeUnit_Min,
            tooltip=False
        )
    elif state.volume_chart: # ignore - wont work
        dpg.add_candle_series(v, o, c, l, h,
            parent="y_axis",
            tag="candles",
            weight=0.5,
            tooltip=False
        )
    else:
        add_text_status(state, "Error: volume_chart state is invalid.")

    if dpg.does_item_exist("EMA_line_series"):
        dpg.delete_item("EMA_line_series")

    
    # ADD LINE SERIES HERE WITH THE EMA

    chart_line_series_x = x[199:]

    chart_line_series_y = state.ema_data_values.iloc[199:].tolist()
     # We need to faze out the first part of the array

    dpg.add_line_series(chart_line_series_x, chart_line_series_y, parent = "y_axis", label = "EMA", tag = "EMA_line_series", skip_nan=True)




    dpg.add_stem_series(x, v, parent="y_axis_indicators", tag="volume_stem")

    if not dpg.does_item_exist("candle_tip"):
        with dpg.tooltip("candles", tag="candle_tip"):
            dpg.add_text("", tag="tip_date")
            dpg.add_text("", tag="tip_open")
            dpg.add_text("", tag="tip_high")
            dpg.add_text("", tag="tip_low")
            dpg.add_text("", tag="tip_close")

    dpg.fit_axis_data("x_axis"); dpg.fit_axis_data("y_axis")

def tooltip_loop(state: AppState):

    cached = None
    while dpg.is_dearpygui_running():
        if dpg.does_item_exist("candles") and dpg.is_item_shown("chart") and state.csv_data is not None: # "is item shown" not working when window closed
            #print(dpg.is_item_shown("candles")) # DEBUG
            if cached is None:
                # Check what chart is loaded: eth & ftse are formatted differently
                # Find the identifier used for the data.
                df = ensure_time_col(state.csv_data)
                df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce") # new
                cached = {
                    
                    "x" : (pd.to_datetime(df["Date"]).values.astype("int64") // 10**9).tolist(),
                    #"x": (df["Date"].astype("int64") // 10**9).tolist(),
                    "o": df["Open"].tolist(),
                    "h": df["High"].tolist(),
                    "l": df["Low"].tolist(),
                    "c": df["Close"].tolist(),
                    "dates": df["Date"].dt.strftime("%d %b %H:%M").tolist()
                }
            if dpg.is_item_hovered("plot"):
                x_pos, _ = dpg.get_plot_mouse_pos()
                if cached["x"]:
                    idx = min(range(len(cached["x"])), key=lambda i: abs(cached["x"][i] - x_pos))
                    dpg.set_value("tip_date",  f"Date:  {cached['dates'][idx]}")
                    dpg.set_value("tip_open",  f"Open:  {cached['o'][idx]}")
                    dpg.set_value("tip_high",  f"High:  {cached['h'][idx]}")
                    dpg.set_value("tip_low",   f"Low:   {cached['l'][idx]}")
                    dpg.set_value("tip_close", f"Close: {cached['c'][idx]}")
                time.sleep(0.05)
            else:
                time.sleep(0.2)
        else:
            cached = None
            #add_text_status(state, "Tooltip loop: waiting for chart...") # DEBUG
            time.sleep(1)

def weight_slider_cb(sender, app_data):
    if dpg.does_item_exist("candles"):
        dpg.configure_item("candles", weight=float(app_data))


def crosshair_cb():
    dpg.configure_item("plot", crosshairs=bool(dpg.get_value("crosshair_value")))

def configure_main_plot_cb():
    if not dpg.is_item_shown("chart"):
        return
    main_h = dpg.get_item_height("chart") - 38
    ratio = dpg.get_value("ratio_change")
    candle_h = int(main_h * ratio)
    vol_h = main_h - candle_h
    dpg.set_item_height("child_window_candle", candle_h)
    dpg.set_item_height("child_window_histo", vol_h)
    dpg.configure_item("ratio_change",  width=int(0.4 * dpg.get_item_width("chart")))
    dpg.configure_item("weight_slider", width=int(0.4 * dpg.get_item_width("chart")))

def sync_on_zoom_cb(sender, app_data):
    if sender == "x_axis":
        x_min, x_max = dpg.get_axis_limits("x_axis")
        dpg.set_axis_limits("x_axis_indicators", x_min, x_max)
    elif sender == "x_axis_indicators":
        x_min, x_max = dpg.get_axis_limits("x_axis_indicators")
        dpg.set_axis_limits("x_axis", x_min, x_max)

def chart_fullsize(state, sender, app_data):
    # Ensure you're actually getting numbers
    if sender == "eth_chart_fullscreen":  # only works if the button/tag is literally this
        # Current chart *displayed* size is usually best read via rect size:
        w_c, h_c = dpg.get_item_rect_size("chart")
        add_text_status(state, f"Chart size is {w_c}×{h_c}")

        # Call the functions!
        height_v = int(dpg.get_viewport_height())
        width_v  = int(dpg.get_viewport_width())

        add_text_status(state, f"Viewport size is {width_v}×{height_v}")

        # Toggle logic: avoid exact equality because of padding/margins
        FULLSCREEN_PAD = 8  # some slack for viewport padding
        is_fullscreen = abs(h_c - height_v) <= FULLSCREEN_PAD and abs(w_c - width_v) <= FULLSCREEN_PAD

        if is_fullscreen:
            dpg.configure_item("chart", height=300, width=400)
            add_text_status(state, "Configuring… SMALLER")
        else:
            # Optionally account for menu bars/status bars etc.
            dpg.configure_item("chart", height=height_v - 0, width=width_v - 0, pos=(0,0))
            add_text_status(state, "Configuring… BIGGER")

def settings_window(state, sender, app_data):
    if sender == "save_and_close":
        # Save settings here
        ema_period = dpg.get_value("ema_period_input")
        state.ema_period = ema_period
        add_text_status(state, f"EMA period set to {ema_period}")
        # Configure the EMA df to reflect the new period
        set_ema_values(state, state.csv_data)
        dpg.delete_item("EMA_line_series") # Remove old line
        volume_state = dpg.get_value("volume_chart_checkbox")
        if volume_state == True:    #edited
            state.volume_chart = True
        else:
            state.volume_chart = False
        generate_chart(state)
        dpg.hide_item("chart_settings")

    else:
        dpg.set_value("ema_period_input", state.ema_period)

        dpg.show_item("chart_settings")
# Edit
def switch_to_volume(state: AppState):
    """Display Volume indicator in the indicator plot"""
    # Clear existing series
    for tag in ("volume_stem", "rsi_line", "roc_line", "macd_line", "signal_line"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    
    if state.csv_data is None:
        add_text_status(state, "No CSV loaded")
        return
    
    df = ensure_time_col(state.csv_data)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    x = (pd.to_datetime(df["Date"]).values.astype("int64") // 10**9).tolist()
    v = df["Volume"].tolist()
    
    dpg.add_stem_series(x, v, parent="y_axis_indicators", tag="volume_stem")
    dpg.configure_item("y_axis_indicators", label="VOLUME")
    dpg.fit_axis_data("x_axis_indicators")
    dpg.fit_axis_data("y_axis_indicators")
    add_text_status(state, "Switched to Volume")

def switch_to_rsi(state: AppState):
    """Display RSI indicator in the indicator plot"""
    # Clear existing series
    for tag in ("volume_stem", "rsi_line", "roc_line", "macd_line", "signal_line"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    
    if state.rsi_values is None:
        add_text_status(state, "RSI data not available")
        return
    
    # Use Date from state.rsi_values to match the length of RSI data
    rsi_dates = pd.to_datetime(state.rsi_values["Date"], utc=True, errors="coerce")
    x = (rsi_dates.values.astype("int64") // 10**9).tolist()
    
    # Get the RSI column (format: rsi_14 or rsi_<period>)
    rsi_col = [col for col in state.rsi_values.columns if col.startswith("rsi_")][0]
    rsi_y = state.rsi_values[rsi_col].tolist()
    
    dpg.add_line_series(x, rsi_y, parent="y_axis_indicators", tag="rsi_line", label="RSI")
    dpg.configure_item("y_axis_indicators", label="RSI")
    dpg.fit_axis_data("x_axis_indicators")
    dpg.fit_axis_data("y_axis_indicators")
    add_text_status(state, "Switched to RSI")

def switch_to_roc(state: AppState):
    """Display ROC indicator in the indicator plot"""
    # Clear existing series
    for tag in ("volume_stem", "rsi_line", "roc_line", "macd_line", "signal_line"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    
    if state.roc_values is None:
        add_text_status(state, "ROC data not available")
        return
    
    # Use Date from state.roc_values to match the length of ROC data
    roc_dates = pd.to_datetime(state.roc_values["Date"], utc=True, errors="coerce")
    x = (roc_dates.values.astype("int64") // 10**9).tolist()
    
    # Get the ROC column (format: roc_12 or roc_<period>)
    roc_col = [col for col in state.roc_values.columns if col.startswith("roc_")][0]
    roc_y = state.roc_values[roc_col].tolist()
    
    dpg.add_line_series(x, roc_y, parent="y_axis_indicators", tag="roc_line", label="ROC")
    dpg.configure_item("y_axis_indicators", label="ROC")
    dpg.fit_axis_data("x_axis_indicators")
    dpg.fit_axis_data("y_axis_indicators")
    add_text_status(state, "Switched to ROC")

def switch_to_macd(state: AppState):
    """Display MACD indicator in the indicator plot"""
    # Clear existing series
    for tag in ("volume_stem", "rsi_line", "roc_line", "macd_line", "signal_line"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    
    if state.macd_values is None:
        add_text_status(state, "MACD data not available")
        return
    
    # Use Date from state.macd_values to match the length of MACD data
    macd_dates = pd.to_datetime(state.macd_values["Date"], utc=True, errors="coerce")
    x = (macd_dates.values.astype("int64") // 10**9).tolist()
    
    # Get the MACD columns (format: macd_12_26_9, macd_signal_12_26_9, etc.)
    macd_col = [col for col in state.macd_values.columns if col.startswith("macd_") and "signal" not in col and "hist" not in col][0]
    signal_col = [col for col in state.macd_values.columns if col.startswith("macd_signal_")][0]
    
    add_text_status(state, state.macd_values.tail().to_string())  # Debug line to check MACD values

    macd_y = state.macd_values[macd_col].tolist()
    signal_y = state.macd_values[signal_col].tolist()
    
    dpg.add_line_series(x, macd_y, parent="y_axis_indicators", tag="macd_line", label="MACD", skip_nan=True)
    dpg.add_line_series(x, signal_y, parent="y_axis_indicators", tag="signal_line", label="Signal", skip_nan=True)
    dpg.configure_item("y_axis_indicators", label="MACD")
    dpg.fit_axis_data("x_axis_indicators")
    dpg.fit_axis_data("y_axis_indicators")
    # Fit the axis data so it fits nicely:

    add_text_status(state, "Switched to MACD")







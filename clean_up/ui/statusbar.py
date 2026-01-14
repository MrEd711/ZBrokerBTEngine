# ui/statusbar.py
import dearpygui.dearpygui as dpg
from state import AppState

def add_text_status(state: AppState, text: str):
    if dpg.is_item_shown("status_bar"):
        dpg.add_text(text, parent="status_child")
        if dpg.get_value("auto_scroll_value"):
            max_scroll = dpg.get_y_scroll_max("status_child") + 20
            dpg.set_y_scroll("status_child", max_scroll)

def configure_status_bar_cb(state: AppState):
    if not dpg.is_item_shown("status_bar"):
        return
    vp_h = dpg.get_viewport_height()
    vp_w = dpg.get_viewport_width()
    # A single, consistent placement rule:
    dpg.configure_item("status_bar",
                       pos=(0, vp_h - state.status_height - 38),
                       width=vp_w - 15,
                       height=state.status_height)

def add_text_status_backtest(state: AppState, text: str):
    if dpg.is_item_shown("backtest_status"):
        dpg.add_text(text, parent="backtest_status")
        if dpg.get_value("auto_scroll_value_backtest"):
            max_scroll = dpg.get_y_scroll_max("backtest_status") + 20
            dpg.set_y_scroll("backtest_status", max_scroll)


def bottom_status_backtest(state: AppState):
    if dpg.is_item_shown("backtest_status"):
        max_scroll = dpg.get_y_scroll_max("backtest_status") + 20
        dpg.set_y_scroll("backtest_status", max_scroll)



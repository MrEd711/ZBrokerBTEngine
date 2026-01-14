import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from state import AppState
from ui.statusbar import add_text_status, add_text_status_backtest
import time as t
import dearpygui.dearpygui as dpg 

def load_data(state: AppState):
    # Load and preprocess data for training

    if state.csv_data is None:
        add_text_status(state, "No CSV data loaded for AI training.")
        add_text_status(state, "No CSV data loaded for AI training.")
        return
    
    whole_data = state.csv_data

    add_text_status(state, "Loading data for AI training...")
    add_text_status(state, whole_data.head().to_string())
    #print(whole_data.head().to_string()) # DEBUG
    add_text_status(state, f"Ready for training on {len(whole_data)} rows.")


def train_model(X, y):
    pass

def create_features(data):
    # Engineering technical features derived from the data
    # Each time stamp is represented by a feature vector including for now: RSI, SMA (Multiple), EMA (Multiple), MACD, VOLUME, ATR 
    pass
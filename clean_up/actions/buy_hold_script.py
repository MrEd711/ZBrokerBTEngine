import pandas as pd
from state import AppState
from ui.statusbar import add_text_status



def buy_and_hold_strategy(state: AppState): # Create percentage returns for buy and hold
    if state.csv_data is None or state.csv_path is None:
        add_text_status(state, "No CSV loaded for backtesting.")
        return

    df = state.csv_data.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Close'])

    df = df.sort_values(by='Date').reset_index(drop=True)

    df['Buy_and_Hold_Returns'] = df['Close'].pct_change() * 100
    df['Cumulative Percentage Returns'] = (1 + df['Buy_and_Hold_Returns'] / 100).cumprod() - 1
    df['Cumulative Percentage Returns'] *= 100

    add_text_status(state, "Buy and Hold data calculated.")
    add_text_status(state, df[['Date', 'Close', 'Buy_and_Hold_Returns', 'Cumulative Percentage Returns']].head().to_string())
    # The difference between normal returns and cumulative returns is that cumulative returns show the total growth over time, while normal returns show the periodic change.
    state.buy_hold = df
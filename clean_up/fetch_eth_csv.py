#!/usr/bin/env python3
"""
Download OHLCV data for a Binance spot symbol and save it as a CSV that chart_viewer.py can read.
"""

import pathlib
import time
from argparse import ArgumentParser
from typing import List
from state import AppState
import requests
import pandas as pd
from pandas_datareader import data as pdr

state = AppState()

BINANCE_SPOT_KLINES = "https://api.binance.com/api/v3/klines"
LIMIT = 1000  # Binance max per request for /api/v3/klines is 1000


def fetch_chunk(symbol: str, start_ms: int, interval: str) -> List[list]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": LIMIT,
    }
    r = requests.get(BINANCE_SPOT_KLINES, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def main(state: AppState):
    p = ArgumentParser()

    p.add_argument(
        "--symbol", "-s",
        default="ETHUSDT",
        help="Binance spot symbol (e.g. ETHUSDT, BTCUSDT)."
    )
    p.add_argument(
        "--days", "-d",
        type=int,
        default=20,
        help="Number of past days to fetch."
    )
    p.add_argument(
        "--interval", "-i",
        default="15m",
        help="Kline interval (e.g. 1m, 5m, 15m, 1h, 1d)."
    )
    p.add_argument(
        "--out", "-o",
        type=pathlib.Path,
        default=None,
        help="Output CSV filename"
    )

    # Option to disable fetching S&P 500 from the UI (pass --no-sp500 to disable)
    p.add_argument(
        "--no-sp500",
        action="store_false",
        dest="fetch_sp500",
        default=True,
        help="Disable fetching S&P 500 data"
    )

    args = p.parse_args()
    symbol = args.symbol.upper().strip()

    # default output file if not provided
    out_path = args.out
    if out_path is None:
        out_path = pathlib.Path(f"{symbol}_{args.interval}.csv")

    # compute ms timestamps
    end_ts_ms = int(time.time() * 1000)
    start_ts_ms = end_ts_ms - args.days * 24 * 60 * 60 * 1000

    # fetch in chunks
    all_rows = []
    fetch_from = start_ts_ms

    while fetch_from < end_ts_ms:
        chunk = fetch_chunk(symbol, fetch_from, args.interval)
        if not chunk:
            break

        all_rows.extend(chunk)

        # CloseTime is index 6; add 1 ms so we don't refetch the last candle
        fetch_from = chunk[-1][6] + 1

        # be polite to the API
        time.sleep(0.35)

        # If Binance returns fewer than LIMIT rows, you've hit the end of available data
        if len(chunk) < LIMIT:
            break

    if not all_rows:
        raise RuntimeError(f"No data returned for symbol={symbol} interval={args.interval}")

    # to DataFrame
    df = pd.DataFrame(
        all_rows,
        columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVolume", "NumberOfTrades",
            "TakerBuyBase", "TakerBuyQuote", "Ignore"
        ],
    )

    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    num_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[num_cols] = df[num_cols].astype(float)

    # save asset data
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")


    if args.fetch_sp500:
        # Fetch S&P 500 when requested
        start_date = df["Date"].iloc[0].strftime("%Y-%m-%d")
        end_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

        print(f"Fetching S&P 500 from {start_date} to {end_date}")
        try:
            sp500 = pdr.DataReader("^SPX", "stooq", start=start_date, end=end_date)
            sp500 = sp500.sort_index()
        except Exception as e:
            print(f"Failed to fetch S&P 500 data: {e}")
            return

        if not sp500.empty:
            sp500 = sp500.reset_index()
            sp500 = sp500[["Date", "Close"]]
            sp500["Close"] = sp500["Close"].astype(float)

            first_close = sp500["Close"].iloc[0]
            sp500["Return_%"] = (sp500["Close"] / first_close - 1.0) * 100

            sp500.to_csv("sp500.csv", index=False)
            print(f"S&P 500 data saved → sp500.csv ({len(sp500)} rows)")
        else:
            print("Failed to fetch S&P 500 data.")


if __name__ == "__main__":
    main(state=AppState())

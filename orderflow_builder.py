"""
orderflow_builder.py  -  step 1 + 2 of the order-flow test

Downloads Binance USD-M futures aggTrades (free public files) for one symbol
and builds, per candle:
  buy_vol / sell_vol  (aggressor volume), delta, CVD, trade count, POC price
plus one volume profile (POC, value area high/low) for the whole period.

Usage (in Codespaces / terminal):
  pip install pandas requests
  python orderflow_builder.py DOTUSDT 2026-09          # one month, 30m candles
  python orderflow_builder.py DOTUSDT 2026-08 2026-09 2026-10 --tf 30min

Output (folder ./orderflow_out):
  DOTUSDT_30min_orderflow.csv   one row per candle
  DOTUSDT_volume_profile.csv    price bins with total volume
  DOTUSDT_profile_summary.txt   POC / value area high / value area low
"""
import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
COLS = ["agg_id", "price", "qty", "first_id", "last_id", "time", "is_buyer_maker"]


def download_month(symbol: str, month: str, cache_dir: str) -> str:
    """Download one monthly aggTrades zip (skips if already downloaded)."""
    os.makedirs(cache_dir, exist_ok=True)
    name = f"{symbol}-aggTrades-{month}.zip"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:

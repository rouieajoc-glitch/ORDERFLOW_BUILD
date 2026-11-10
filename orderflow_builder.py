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
        print(f"  already have {name}")
        return path
    url = f"{BASE}/{symbol}/{name}"
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code == 404:
            print(f"Not found: {url}")
            print("Monthly files appear on the first Monday of the next month. Try an older month.")
            sys.exit(1)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return path


def read_zip(path: str) -> pd.DataFrame:
    """Read the CSV inside the zip. Handles files with or without a header row."""
    with zipfile.ZipFile(path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as f:
            raw = f.read()
    head = raw[:200].decode("utf-8", errors="ignore").splitlines()[0]
    has_header = head.lower().startswith("agg_trade_id") or not head[0].isdigit()
    df = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else COLS,
        usecols=None,
    )
    df.columns = COLS
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().eq("true")
    # time is milliseconds for futures; guard for microseconds just in case
    t = df["time"].astype("int64")
    unit = "us" if t.iloc[0] > 10**14 else "ms"
    df["time"] = pd.to_datetime(t, unit=unit, utc=True)
    return df[["time", "price", "qty", "is_buyer_maker"]]


def build_candles(df: pd.DataFrame, tf: str, tick: float) -> pd.DataFrame:
    """Aggressor buy = buyer was NOT the maker. Aggressor sell = buyer was the maker."""
    df = df.copy()
    df["buy"] = np.where(~df["is_buyer_maker"], df["qty"], 0.0)
    df["sell"] = np.where(df["is_buyer_maker"], df["qty"], 0.0)
    df["bin"] = (df["price"] / tick).round().astype("int64")
    df["candle"] = df["time"].dt.floor(tf)

    g = df.groupby("candle")
    out = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "buy_vol": g["buy"].sum(),
            "sell_vol": g["sell"].sum(),
            "trades": g["qty"].size(),
        }
    )
    out["delta"] = out["buy_vol"] - out["sell_vol"]
    out["cvd"] = out["delta"].cumsum()

    # POC per candle: price bin with the most volume inside that candle
    vol_by_bin = df.groupby(["candle", "bin"])["qty"].sum()
    poc_bin = vol_by_bin.groupby(level=0).idxmax().map(lambda ix: ix[1])
    out["poc"] = (poc_bin * tick).round(8)
    return out.reset_index()


def volume_profile(df: pd.DataFrame, tick: float, va_pct: float = 0.70):
    """Whole-period volume profile: POC and value area (default 70% of volume)."""
    bins = (df["price"] / tick).round().astype("int64")
    prof = df.groupby(bins)["qty"].sum().sort_index()
    prof.index = (prof.index * tick).round(8)
    poc = prof.idxmax()

    # grow the value area outward from the POC, adding the larger neighbour each step
    prices = prof.index.to_list()
    vols = prof.to_list()
    i = prices.index(poc)
    lo = hi = i
    total = sum(vols)
    acc = vols[i]
    while acc < va_pct * total and (lo > 0 or hi < len(prices) - 1):
        up = vols[hi + 1] if hi < len(prices) - 1 else -1
        dn = vols[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1
            acc += vols[hi]
        else:
            lo -= 1
            acc += vols[lo]
    return prof, poc, prices[hi], prices[lo]


def guess_tick(df: pd.DataFrame) -> float:
    """Pick a price bin size: about 0.02% of the median price, rounded to a clean value."""
    med = float(df["price"].median())
    raw = med * 0.0002
    mag = 10 ** np.floor(np.log10(raw))
    return round(float(round(raw / mag) * mag), 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("months", nargs="+", help="e.g. 2026-09 2026-10")
    ap.add_argument("--tf", default="30min", help="candle size, e.g. 5min, 30min, 1h")
    ap.add_argument("--tick", type=float, default=0.0, help="price bin size (0 = auto)")
    ap.add_argument("--out", default="orderflow_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames = []
    for m in args.months:
        print(f"{args.symbol} {m}")
        path = download_month(args.symbol, m, os.path.join(args.out, "raw"))
        frames.append(read_zip(path))
    df = pd.concat(frames, ignore_index=True).sort_values("time")
    print(f"loaded {len(df):,} aggregated trades")

    tick = args.tick or guess_tick(df)
    print(f"price bin size: {tick}")

    candles = build_candles(df, args.tf, tick)
    prof, poc, vah, val = volume_profile(df, tick)

    candles.to_csv(os.path.join(args.out, f"{args.symbol}_{args.tf}_orderflow.csv"), index=False)
    prof.rename("volume").rename_axis("price").to_csv(
        os.path.join(args.out, f"{args.symbol}_volume_profile.csv")
    )
    with open(os.path.join(args.out, f"{args.symbol}_profile_summary.txt"), "w") as f:
        f.write(f"POC {poc}\nValue area high {vah}\nValue area low {val}\n")

    print(f"candles: {len(candles)}   POC {poc}   VAH {vah}   VAL {val}")
    print(f"saved to ./{args.out}")


if __name__ == "__main__":
    main()

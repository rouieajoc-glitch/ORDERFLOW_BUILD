"""
orderflow_test.py  -  step 3: does a delta filter improve 15m OB retests?

Needs the 15m candle file made by orderflow_builder.py:
  python orderflow_builder.py DOTUSDT 2026-07 2026-08 2026-09 --tf 15min
Then:
  python orderflow_test.py DOTUSDT

SIMPLIFIED rules (not identical to the TradingView indicator):
  Bullish OB  = last bearish candle (within 5 bars) before a bullish displacement
                candle (body >= DISP x ATR) that closes above the OB candle's high.
  Retest      = FIRST later candle whose low comes within NEAR x ATR of the OB high.
                If it closes below the OB low, the OB is dead and no trade is taken.
  Entry       = next candle's open.  Stop = beyond the OB low (STOPBUF x ATR).
  Target      = 2R and 3R tested separately.  Timeout = MAXHOLD bars, exit at close.
  If stop and target are both hit inside one candle, the stop is assumed first.
  Shorts are the mirror image.
  Costs       = (FEE + SLIP) per side, charged on entry and exit, converted into R.

Delta filter: delta ratio of the retest candle = (buy - sell) / (buy + sell)
  ALIGNED  = ratio > 0 for longs, < 0 for shorts
  AGAINST  = the opposite
Results are shown for ALL / ALIGNED / AGAINST, longs and shorts separately.
"""
import argparse
import os

import numpy as np
import pandas as pd


def add_atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    df["atr"] = tr.rolling(n).mean().shift(1)  # shifted: no look-ahead
    return df


def find_obs(df, disp, lookback=5):
    """Return list of (side, ob_index, displacement_index, zone_low, zone_high)."""
    o, c, h, l, atr = (df[k].to_numpy() for k in ("open", "close", "high", "low", "atr"))
    obs, used = [], set()
    for d in range(20, len(df)):
        if np.isnan(atr[d]):
            continue
        body = abs(c[d] - o[d])
        if body < disp * atr[d]:
            continue
        if c[d] > o[d]:  # bullish displacement -> bullish OB
            for j in range(d - 1, max(d - lookback - 1, 0), -1):
                if c[j] < o[j]:
                    if c[d] > h[j] and ("L", j) not in used:
                        used.add(("L", j))
                        obs.append(("L", j, d, l[j], h[j]))
                    break
        else:  # bearish displacement -> bearish OB
            for j in range(d - 1, max(d - lookback - 1, 0), -1):
                if c[j] > o[j]:
                    if c[d] < l[j] and ("S", j) not in used:
                        used.add(("S", j))
                        obs.append(("S", j, d, l[j], h[j]))
                    break
    return obs


def simulate(df, obs, near, stopbuf, maxage, maxhold, targets, cost_pct):
    o, c, h, l, atr = (df[k].to_numpy() for k in ("open", "close", "high", "low", "atr"))
    buy, sell = df["buy_vol"].to_numpy(), df["sell_vol"].to_numpy()
    times = df["candle"].to_numpy()
    rows = []
    n = len(df)
    for side, j, d, zlo, zhi in obs:
        retest = None
        for t in range(d + 2, min(d + maxage, n - 2)):
            if side == "L":
                if c[t] < zlo:
                    break
                if l[t] <= zhi + near * atr[t]:
                    retest = t if c[t] > zlo else None
                    break
            else:
                if c[t] > zhi:
                    break
                if h[t] >= zlo - near * atr[t]:
                    retest = t if c[t] < zhi else None
                    break
        if retest is None:
            continue
        t = retest
        e = t + 1
        entry = o[e]
        if side == "L":
            stop = zlo - stopbuf * atr[t]
            risk = entry - stop
        else:
            stop = zhi + stopbuf * atr[t]
            risk = stop - entry
        if risk < 0.3 * atr[t]:
            continue
        tot = buy[t] + sell[t]
        ratio = (buy[t] - sell[t]) / tot if tot > 0 else 0.0
        cost_r = cost_pct * entry / risk
        rec = {"time": times[t], "side": side, "entry": entry, "risk_pct": risk / entry * 100,
               "delta_ratio": ratio}
        for R in targets:
            gross = None
            for k in range(e, min(e + maxhold, n)):
                if side == "L":
                    if l[k] <= stop:
                        gross = -1.0
                        break
                    if h[k] >= entry + R * risk:
                        gross = float(R)
                        break
                else:
                    if h[k] >= stop:
                        gross = -1.0
                        break
                    if l[k] <= entry - R * risk:
                        gross = float(R)
                        break
            if gross is None:  # timeout
                last = min(e + maxhold, n) - 1
                gross = ((c[last] - entry) if side == "L" else (entry - c[last])) / risk
            rec[f"net_{R}R"] = gross - cost_r
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(tr, R):
    col = f"net_{R}R"
    if len(tr) == 0:
        return "n=0"
    x = tr[col]
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return (f"n={len(x):3d}  win={100*(x>0).mean():4.0f}%  "
            f"avgR={x.mean():+.2f} (+/-{se:.2f})  totalR={x.sum():+.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--dir", default="orderflow_out")
    ap.add_argument("--tf", default="15min")
    ap.add_argument("--disp", type=float, default=1.5, help="displacement body x ATR")
    ap.add_argument("--near", type=float, default=0.3, help="retest distance x ATR")
    ap.add_argument("--stopbuf", type=float, default=0.1, help="stop buffer x ATR")
    ap.add_argument("--maxage", type=int, default=192, help="bars an OB stays valid")
    ap.add_argument("--maxhold", type=int, default=192, help="max bars in trade")
    ap.add_argument("--fee", type=float, default=0.0005, help="fee per side (0.0005 = 0.05%%)")
    ap.add_argument("--slip", type=float, default=0.0005, help="slippage per side")
    ap.add_argument("--minratio", type=float, default=0.0, help="delta ratio threshold")
    args = ap.parse_args()

    path = os.path.join(args.dir, f"{args.symbol}_{args.tf}_orderflow.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Build it first with --tf {args.tf}.")
    df = pd.read_csv(path, parse_dates=["candle"])
    df = add_atr(df)
    obs = find_obs(df, args.disp)
    targets = (2, 3)
    cost_pct = 2 * (args.fee + args.slip)
    tr = simulate(df, obs, args.near, args.stopbuf, args.maxage, args.maxhold, targets, cost_pct)
    print(f"{len(df)} candles, {len(obs)} OBs found, {len(tr)} retest trades, "
          f"round-trip cost {100*cost_pct:.2f}%")
    if len(tr) == 0:
        return
    tr.to_csv(os.path.join(args.dir, f"{args.symbol}_ob_trades.csv"), index=False)

    for side, name in (("L", "LONGS"), ("S", "SHORTS")):
        t = tr[tr["side"] == side]
        sign = 1 if side == "L" else -1
        aligned = t[sign * t["delta_ratio"] > args.minratio]
        against = t[sign * t["delta_ratio"] < -args.minratio]
        print(f"\n{name}")
        for R in targets:
            print(f"  {R}R target")
            print(f"    ALL      {summarize(t, R)}")
            print(f"    ALIGNED  {summarize(aligned, R)}")
            print(f"    AGAINST  {summarize(against, R)}")

    print("\nHow to read this: avgR is after costs. '+/-' is the standard error; if the "
          "ALIGNED and ALL ranges overlap, the filter has shown nothing yet. "
          "Fewer than ~30 trades per row means treat it as noise.")


if __name__ == "__main__":
    main()

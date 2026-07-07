"""Probe MT5 for correlated-asset symbols and fetch M5 over gold's span
for SMT-divergence research. Caches to data/lab_<sym>_cache.csv.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.mt5_client import MT5Client
from src.research import mtf

# Gold's cached span (broker time)
gold = mtf.load_m5()
g0, g1 = gold["timestamp"].iloc[0], gold["timestamp"].iloc[-1]
print(f"Gold span: {g0} -> {g1}  ({len(gold)} M5 bars)")

# Candidate correlated symbols, priority order. Broker suffixes vary, so try
# a few spellings for each concept.
CANDIDATES = {
    "XAGUSD": ["XAGUSD", "XAGUSD.", "SILVER", "Silver"],
    "EURUSD": ["EURUSD", "EURUSD."],
    "DXY":    ["DXY", "USDX", "USDIDX", "USDOLLAR", "US Dollar Index"],
}

client = MT5Client()
if not client.connect():
    print("MT5 connect failed")
    sys.exit(1)

import MetaTrader5 as mt5
found = {}
for concept, names in CANDIDATES.items():
    for nm in names:
        info = mt5.symbol_info(nm)
        if info is not None:
            found[concept] = nm
            print(f"  {concept:8s} -> available as '{nm}'")
            break
    else:
        print(f"  {concept:8s} -> NOT available (tried {names})")

import time
start = datetime(g0.year, g0.month, g0.day)
end = pd.Timestamp(g1)
data_dir = Path(__file__).resolve().parents[1] / "data"

for concept, sym in found.items():
    # Ensure symbol is in Market Watch and give history a moment to load
    mt5.symbol_select(sym, True)
    time.sleep(1.0)
    # copy_rates_from: `count` bars starting AT start going forward — reliably
    # reaches back to 2024-09 (count-from-pos anchors at "now" and misses it).
    rates = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M5, start, 160000)
    if rates is None or len(rates) == 0:
        print(f"  {concept}: copy_rates_from returned nothing ({mt5.last_error()})")
        continue
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df["volume"] = df.get("tick_volume", 0)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df[df["timestamp"] <= end]
    df = df.sort_values("timestamp").reset_index(drop=True)
    path = data_dir / f"lab_{concept.lower()}_cache.csv"
    df.to_csv(path, index=False)
    # alignment check vs gold
    merged = pd.merge(gold[["timestamp"]], df[["timestamp"]], on="timestamp", how="inner")
    print(f"  {concept}: {len(df)} bars {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} | "
          f"{len(merged)}/{len(gold)} timestamps align with gold ({len(merged)/len(gold)*100:.0f}%) -> {path.name}")

client.disconnect() if hasattr(client, "disconnect") else None
print("done")

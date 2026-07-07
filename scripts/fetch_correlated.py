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

start = datetime(g0.year, g0.month, g0.day)
end = datetime(g1.year, g1.month, g1.day, 23, 59)
data_dir = Path(__file__).resolve().parents[1] / "data"

for concept, sym in found.items():
    df = client.get_candles(sym, "M5", start_date=start, end_date=end)
    if df is None or df.empty:
        print(f"  {concept}: fetch returned nothing")
        continue
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    path = data_dir / f"lab_{concept.lower()}_cache.csv"
    df.to_csv(path, index=False)
    # alignment check vs gold
    merged = pd.merge(gold[["timestamp"]], df[["timestamp"]], on="timestamp", how="inner")
    print(f"  {concept}: {len(df)} bars {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} | "
          f"{len(merged)}/{len(gold)} timestamps align with gold ({len(merged)/len(gold)*100:.0f}%) -> {path.name}")

client.disconnect() if hasattr(client, "disconnect") else None
print("done")

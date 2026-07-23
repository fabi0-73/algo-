"""Feasibility probe: how deep and rich is our broker's XAUUSD tick history?

Read-only MT5 queries. Determines whether a tick-microstructure program
(cost calibration + AMD entry conditioning) is viable and over what span.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5

from src.data.mt5_client import MT5Client

client = MT5Client()
if not client.connect():
    print("MT5 connect failed")
    sys.exit(1)

now = datetime.now(timezone.utc)
print(f"probe at {now:%Y-%m-%d %H:%M} UTC")

# binary-search-ish probe: try increasingly old 1-hour windows
for days_back in (1, 7, 30, 60, 90, 180, 365, 730):
    t0 = now - timedelta(days=days_back)
    ticks = mt5.copy_ticks_range("XAUUSD", t0, t0 + timedelta(hours=1),
                                 mt5.COPY_TICKS_ALL)
    n = 0 if ticks is None else len(ticks)
    print(f"  {days_back:4d}d back: {n:6d} ticks in 1h window", end="")
    if n:
        t = ticks[0]
        print(f"  | fields: time_msc={t['time_msc']} bid={t['bid']} "
              f"ask={t['ask']} last={t['last']} vol={t['volume']} "
              f"flags={t['flags']}", end="")
    print()

# density check on the most recent complete hour
t0 = now - timedelta(hours=2)
ticks = mt5.copy_ticks_range("XAUUSD", t0, t0 + timedelta(hours=1),
                             mt5.COPY_TICKS_ALL)
if ticks is not None and len(ticks):
    import numpy as np
    bid = ticks["bid"]; ask = ticks["ask"]
    spread = ask - bid
    print(f"\nrecent hour: {len(ticks)} ticks | spread mean {spread.mean():.3f} "
          f"p95 {np.percentile(spread, 95):.3f} max {spread.max():.3f} | "
          f"last-price nonzero: {(ticks['last'] > 0).mean()*100:.0f}% | "
          f"volume nonzero: {(ticks['volume'] > 0).mean()*100:.0f}%")
client.disconnect()

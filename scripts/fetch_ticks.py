"""Backfill XAUUSD quote ticks from MT5 into monthly parquet files.

Quote-only feed (bid/ask/time_msc/flags — no trade tape on this broker).
Fetches day-by-day to bound memory, writes data/ticks/XAUUSD_YYYY-MM.parquet.
Existing months are skipped unless --refresh-last (re-pulls the newest month).

Usage: python scripts/fetch_ticks.py --months 12
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MetaTrader5 as mt5
import pandas as pd

from src.data.mt5_client import MT5Client

OUT = Path(__file__).resolve().parents[1] / "data" / "ticks"
SYMBOL = "XAUUSD"


def month_starts(months_back: int):
    today = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0,
                                               second=0, microsecond=0)
    out = []
    y, m = today.year, today.month
    for _ in range(months_back + 1):
        out.append(datetime(y, m, 1, tzinfo=timezone.utc))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return sorted(out)


def fetch_month(start: datetime) -> pd.DataFrame:
    nxt = (start + timedelta(days=32)).replace(day=1)
    frames = []
    day = start
    while day < min(nxt, datetime.now(timezone.utc)):
        ticks = mt5.copy_ticks_range(SYMBOL, day, day + timedelta(days=1),
                                     mt5.COPY_TICKS_ALL)
        if ticks is not None and len(ticks):
            df = pd.DataFrame(ticks)
            frames.append(df[["time_msc", "bid", "ask", "flags"]])
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="time_msc", keep="first")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--refresh-last", action="store_true")
    args = ap.parse_args()

    client = MT5Client()
    if not client.connect():
        print("MT5 connect failed")
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)

    total = 0
    for start in month_starts(args.months):
        tag = f"{start:%Y-%m}"
        dest = OUT / f"{SYMBOL}_{tag}.parquet"
        is_last = start.month == datetime.now(timezone.utc).month and \
            start.year == datetime.now(timezone.utc).year
        if dest.exists() and not (is_last and args.refresh_last):
            print(f"{tag}: cached ({dest.stat().st_size/1e6:.0f} MB)")
            continue
        df = fetch_month(start)
        if df.empty:
            print(f"{tag}: no ticks served")
            continue
        df.to_parquet(dest, index=False)
        total += len(df)
        print(f"{tag}: {len(df):,} ticks -> {dest.name} "
              f"({dest.stat().st_size/1e6:.0f} MB)")
    print(f"backfill complete: {total:,} new ticks")
    client.disconnect()


if __name__ == "__main__":
    main()

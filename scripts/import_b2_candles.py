"""Import the user-downloaded B2 gold M5 snapshot into Postgres as XAUUSD_B2.

Quarantined symbol name — NEVER mixed with our own broker's XAUUSD rows.
Local files -> local DB only; no network, no credentials.

Usage: python scripts/import_b2_candles.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.db import Database

SNAP = Path(__file__).resolve().parents[1] / "data" / "b2_snapshot"
SYMBOL = "XAUUSD_B2"
CHUNK = 20_000


def main():
    df = pd.read_parquet(SNAP / "XAUUSD_M5_x100screen_v1.parquet")
    df = df.rename(columns={"time": "timestamp"})
    df["symbol"] = SYMBOL
    df["timeframe"] = "M5"
    df["volume"] = df["tick_volume"].astype("int64")
    df = df[["symbol", "timeframe", "timestamp",
             "open", "high", "low", "close", "volume"]]
    print(f"importing {len(df)} rows as {SYMBOL} M5 "
          f"({df.timestamp.min()} -> {df.timestamp.max()})")

    db = Database()
    total = 0
    for i in range(0, len(df), CHUNK):
        total += db.insert_candles(df.iloc[i:i + CHUNK])
        if (i // CHUNK) % 5 == 0:
            print(f"  {i + CHUNK:>7,} / {len(df):,}")
    print(f"done: {total} rows affected")
    print("range in DB:", db.get_date_range(SYMBOL, "M5"))


if __name__ == "__main__":
    main()

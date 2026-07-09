"""
Export candles from Postgres to the portable research cache.

Run on the machine that has the DB (the MT5/fetch box); the cache file is
what every research script consumes elsewhere (mtf.load_m5 / --cache flags).

Usage:
    python scripts/export_cache.py                       # full XAUUSD M5
    python scripts/export_cache.py --gzip                # data/lab_m5_cache.csv.gz
    python scripts/export_cache.py --start 2021-01-01 --out data/lab_m5_cache.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--timeframe", default="M5")
    ap.add_argument("--out", default="data/lab_m5_cache.csv")
    ap.add_argument("--gzip", action="store_true",
                    help="write .csv.gz (pandas-native compression)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    from src.data.db import Database
    db = Database()
    df = db.get_candles(args.symbol, args.timeframe, args.start, args.end)
    if df.empty:
        print(f"No {args.symbol} {args.timeframe} candles in database.")
        sys.exit(1)

    df = df[COLUMNS].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    out = Path(args.out)
    if args.gzip and not str(out).endswith(".gz"):
        out = Path(str(out) + ".gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{len(df)} bars {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("next: python scripts/audit_data.py --cache", out)


if __name__ == "__main__":
    main()

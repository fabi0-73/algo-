"""Build the 8-year research cache from the user-downloaded B2 snapshots.

Reads data/b2_snapshot/XAUUSD_M5_x100screen_v1.parquet (2018-2026 gold M5)
and XAGUSD_M5_long.parquet (silver M5), merges silver close as `xag_close`
(ffill-limited to 3 bars), and writes data/b2_xau_m5_cache.csv.gz in the
research-cache format used by src/research/mtf.load_m5.

Local transformation only — no network, no credentials.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "data" / "b2_snapshot"
OUT = ROOT / "data" / "b2_xau_m5_cache.csv.gz"


def main():
    xau = pd.read_parquet(SNAP / "XAUUSD_M5_x100screen_v1.parquet")
    xag = pd.read_parquet(SNAP / "XAGUSD_M5_long.parquet")
    xau = xau.rename(columns={"time": "timestamp"})
    xag = (xag.rename(columns={"time": "timestamp"})[["timestamp", "close"]]
              .rename(columns={"close": "xag_close"})
              .sort_values("timestamp"))
    print(f"xau: {len(xau)} bars {xau.timestamp.min()} -> {xau.timestamp.max()}")
    print(f"xag: {len(xag)} bars {xag.timestamp.min()} -> {xag.timestamp.max()}")

    df = xau[["timestamp", "open", "high", "low", "close",
              "tick_volume", "spread"]].copy()
    df["volume"] = df["tick_volume"]
    df = df.merge(xag, on="timestamp", how="left")
    df["xag_close"] = df["xag_close"].ffill(limit=3)
    print(f"xag coverage on xau bars: {df['xag_close'].notna().mean()*100:.1f}%")

    df.to_csv(OUT, index=False, compression="gzip")
    print(f"wrote {OUT}  {df.shape}")


if __name__ == "__main__":
    main()

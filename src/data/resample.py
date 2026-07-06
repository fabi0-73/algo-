"""OHLCV resampling from base M5 candles to higher run-timeframes.

The DB stores native M5 (broker time, open-time-stamped bars — MT5 convention).
Higher-timeframe runs load M5 and aggregate here before the engine sees the data,
so one dataset serves every run timeframe and stays in the broker time frame.
"""

import pandas as pd

# Supported run timeframes and their bar interval in minutes
TIMEFRAME_MINUTES = {
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregate M5 candles into the target timeframe.

    Bars are labeled by open time (label='left', closed='left'), matching the
    MT5 convention of the source data, so all downstream hour-of-day logic
    (sessions, news blackouts, rollover, prime hours) keeps its meaning.
    Empty buckets (weekends, gaps) are dropped rather than forward-filled.
    """
    if timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported run timeframe: {timeframe}")
    minutes = TIMEFRAME_MINUTES[timeframe]
    if minutes == 5:
        return df.copy()

    out = df.set_index("timestamp").resample(
        f"{minutes}min", label="left", closed="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    if "symbol" in df.columns and len(df):
        out["symbol"] = df["symbol"].iloc[0]
    return out

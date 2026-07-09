"""Context labels: bucket correctness, <=5 levels + na, lookahead safety of
the H1 trend column."""
import numpy as np
import pandas as pd

from src.research.context import CONTEXT_COLUMNS, context_frame
from src.research.mtf import prepare_frame


def m5_frame(start="2025-01-06 01:00", periods=600, seed=2):
    rng = np.random.default_rng(seed)
    close = 100 + rng.normal(0, 0.2, periods).cumsum()
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    df = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=periods, freq="5min"),
        "open": open_,
        "high": np.maximum(open_, close) + 0.1,
        "low": np.minimum(open_, close) - 0.1,
        "close": close,
        "volume": 100,
    })
    return df


def test_columns_and_levels():
    m5 = m5_frame()
    df = prepare_frame(m5, "M5")
    ctx = context_frame(df, m5)
    assert list(ctx.columns) == CONTEXT_COLUMNS
    assert len(ctx) == len(df)
    for col in CONTEXT_COLUMNS:
        levels = set(ctx[col].unique())
        assert len(levels) <= 6, f"{col} has too many levels: {levels}"


def test_session_buckets():
    m5 = m5_frame()
    df = prepare_frame(m5, "M5")
    ctx = context_frame(df, m5)
    by_hour = dict(zip(df["timestamp"], ctx["session"]))
    assert by_hour[pd.Timestamp("2025-01-06 03:00")] == "asia"
    assert by_hour[pd.Timestamp("2025-01-06 12:00")] == "london"
    assert by_hour[pd.Timestamp("2025-01-06 17:00")] == "ny"
    assert by_hour[pd.Timestamp("2025-01-06 01:00")] == "off"


def test_dow_and_pd_side():
    m5 = m5_frame(periods=600)  # spans into Tue
    df = prepare_frame(m5, "M5")
    ctx = context_frame(df, m5)
    assert ctx.loc[0, "dow"] == "0"  # Monday
    # first day has no prior day -> na
    day0 = df["timestamp"].dt.normalize() == pd.Timestamp("2025-01-06")
    assert (ctx.loc[day0, "pd_side"] == "na").all()
    day1 = ~day0
    assert set(ctx.loc[day1, "pd_side"]) <= {"above", "below"}


def test_h1_trend_no_lookahead():
    m5 = m5_frame(periods=400)
    df = prepare_frame(m5, "M5")
    full = context_frame(df, m5)
    cut = 300
    m5p = m5.iloc[: cut + 1].reset_index(drop=True)
    dfp = prepare_frame(m5p, "M5")
    part = context_frame(dfp, m5p)
    assert full.loc[cut, "h1_trend"] == part.loc[cut, "h1_trend"]
    assert full.loc[cut, "atr_regime"] == part.loc[cut, "atr_regime"]


def test_vwap_side_matches_sign():
    m5 = m5_frame()
    df = prepare_frame(m5, "M5")
    ctx = context_frame(df, m5)
    from src.research.strategies.base import anchored_vwap
    vwap = anchored_vwap(df)
    above = ctx["vwap_side"] == "above"
    assert (df.loc[above, "close"] > vwap[above]).all()

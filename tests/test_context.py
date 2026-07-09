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


def silver_frame(start="2025-01-06 01:00", periods=200, jump_at=None, seed=9):
    """Synthetic silver M5: flat noise, optional +5.0 jump on the close of
    the bar at positional index jump_at (a huge up-momentum event)."""
    rng = np.random.default_rng(seed)
    # noise tiny vs the wick-driven ATR (~0.1) so momentum buckets stay
    # "flat" except for the planted jump
    close = 30 + rng.normal(0, 0.001, periods).cumsum()
    if jump_at is not None:
        close[jump_at:] += 5.0
    open_ = np.roll(close, 1)
    open_[0] = 30.0
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=periods, freq="5min"),
        "open": open_,
        "high": np.maximum(open_, close) + 0.05,
        "low": np.minimum(open_, close) - 0.05,
        "close": close,
        "volume": 50,
    })


def test_xag_mom_prefix_invariant():
    m5 = m5_frame(periods=400)
    df = prepare_frame(m5, "M5")
    xag = silver_frame(periods=400)
    full = context_frame(df, m5, xag=xag)
    cut = 300
    m5p = m5.iloc[: cut + 1].reset_index(drop=True)
    dfp = prepare_frame(m5p, "M5")
    part = context_frame(dfp, m5p, xag=xag)
    assert (full.loc[: cut, "xag_mom"] == part["xag_mom"]).all()


def test_xag_mom_uses_only_closed_silver_bars():
    # Silver jumps +5 on the bar stamped T. That bar closes at T+5, so the
    # gold bar stamped T (evaluated at ITS close, also T+5) must NOT see the
    # jump; the gold bar stamped T+5 must.
    n = 100
    m5 = m5_frame(periods=n + 2)
    df = prepare_frame(m5, "M5")
    xag = silver_frame(periods=n, jump_at=n - 1)
    ctx = context_frame(df, m5, xag=xag)
    t_jump = xag["timestamp"].iloc[-1]
    at_jump = df.index[df["timestamp"] == t_jump][0]
    after = df.index[df["timestamp"] == t_jump + pd.Timedelta(minutes=5)][0]
    assert ctx.loc[at_jump, "xag_mom"] == "flat"  # jump bar not yet closed
    assert ctx.loc[after, "xag_mom"] == "up"


def test_xag_mom_missing_and_stale_silver_are_na():
    m5 = m5_frame(periods=300)
    df = prepare_frame(m5, "M5")
    # empty/absent silver -> all na
    ctx = context_frame(df, m5, xag=pd.DataFrame())
    assert (ctx["xag_mom"] == "na").all()
    # silver covering only the first 100 bars: gold bars beyond the staleness
    # tolerance must be na, never carrying old momentum forward
    xag = silver_frame(periods=100)
    ctx2 = context_frame(df, m5, xag=xag)
    silver_end = xag["timestamp"].iloc[-1]
    stale = df["timestamp"] > silver_end + pd.Timedelta(minutes=35)
    assert (ctx2.loc[stale, "xag_mom"] == "na").all()
    live = df["timestamp"].between(
        silver_end - pd.Timedelta(minutes=60), silver_end)
    assert (ctx2.loc[live, "xag_mom"] != "na").any()

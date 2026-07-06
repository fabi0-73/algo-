"""Tests for the liquidity-sweep entry model (liquidity_levels + sweep_entry)."""
import numpy as np
import pandas as pd
import pytest

from src.strategy.liquidity_levels import (
    LiquidityLevel,
    add_asian_range,
    find_swing_points,
    _cluster_levels,
    get_active_levels,
)
from src.strategy.sweep_entry import detect_sweep_at_candle


BASE_CFG = {
    "use_pdh_pdl": True, "use_weekly": True, "use_asian_range": True,
    "use_round_numbers": True, "round_step": 25.0,
    "use_equal_levels": True, "equal_tolerance_atr_mult": 0.10,
    "equal_min_touches": 2, "level_lookback": 200, "swing_strength": 3,
    "min_poke_atr_mult": 0.10, "max_candles_back_inside": 3,
    "require_rejection": True, "max_level_distance_atr": 1.5,
    "volume_bonus": True, "volume_bonus_ratio": 1.5,
}


class TestAsianRange:
    def test_range_computed_and_masked(self):
        # Two days of hourly candles in the broker frame (asian = 23:00 -> 08:00)
        ts = pd.date_range("2025-01-06 01:00", periods=48, freq="h")
        df = pd.DataFrame({
            "timestamp": ts,
            "open": 2000.0, "close": 2000.0,
            "high": 2001.0, "low": 1999.0,
        })
        # Make the asian window of day 1 (01:00-07:00) distinctive
        asian_mask = np.array((ts.hour >= 23) | (ts.hour < 8))
        day1_mask = np.array(ts.date == ts[0].date())
        df.loc[asian_mask & day1_mask, "high"] = 2010.0
        df.loc[asian_mask & day1_mask, "low"] = 1990.0

        out = add_asian_range(df)
        # A mid-day row on day 1 sees the completed asian range
        day1_noon = out[(pd.to_datetime(out.timestamp).dt.hour == 12)].iloc[0]
        assert day1_noon["asian_high"] == 2010.0
        assert day1_noon["asian_low"] == 1990.0
        # Rows inside the asian session itself have no (forming) range
        inside = out[(pd.to_datetime(out.timestamp).dt.hour == 3)].iloc[0]
        assert pd.isna(inside["asian_high"])

    def test_post_2300_candles_belong_to_next_session(self):
        ts = pd.date_range("2025-01-06 20:00", periods=20, freq="h")
        df = pd.DataFrame({
            "timestamp": ts, "open": 2000.0, "close": 2000.0,
            "high": 2001.0, "low": 1999.0,
        })
        df.loc[pd.to_datetime(df.timestamp).dt.hour == 23, "high"] = 2050.0
        out = add_asian_range(df)
        # The 23:00 spike on Jan 6 must appear in Jan 7's asian_high
        day2_noon = out[(pd.to_datetime(out.timestamp).dt.day == 7)
                        & (pd.to_datetime(out.timestamp).dt.hour == 12)].iloc[0]
        assert day2_noon["asian_high"] == 2050.0


class TestSwingsAndClusters:
    def test_find_swing_points(self):
        highs = np.array([1, 2, 3, 10, 3, 2, 1, 2, 3, 11, 3, 2, 1], dtype=float)
        lows = highs - 1
        sh, sl = find_swing_points(highs, lows, strength=3)
        assert 3 in sh and 9 in sh
        assert len(sl) == 0 or all(lows[i] < lows[i - 1] for i in sl)

    def test_cluster_levels(self):
        prices = np.array([2000.0, 2000.3, 2000.1, 2050.0])
        levels = _cluster_levels(prices, tolerance=0.5, min_touches=2)
        assert len(levels) == 1
        assert abs(levels[0] - 2000.13) < 0.1  # mean of the 3-touch cluster

    def test_round_numbers_and_dedupe(self):
        row_levels = {"prev_day_high": 2025.0, "prev_day_low": 1980.0,
                      "prev_week_high": None, "prev_week_low": None,
                      "asian_high": None, "asian_low": None}
        levels = get_active_levels(300, close=2020.0, atr=2.0,
                                   row_levels=row_levels, cfg=BASE_CFG)
        kinds = {l.kind for l in levels}
        # PDH (2025) and ROUND above (2025) coincide -> merged kind
        assert any("PDH" in k and "ROUND" in k for k in kinds)
        assert any(l.price == 2000.0 and l.side == "BELOW" for l in levels)  # round below


def _mk_arrays(prices_high, prices_low, closes, opens=None, vol=None):
    n = len(closes)
    return (np.array(prices_high, dtype=float), np.array(prices_low, dtype=float),
            np.array(opens if opens is not None else closes, dtype=float),
            np.array(closes, dtype=float),
            np.array(vol if vol is not None else [100] * n, dtype=float))


class TestSweepDetection:
    def test_short_after_sweep_of_highs(self):
        # Level at 2010 (side ABOVE). Bar 3 pokes to 2011 (poke=1.0 >= 0.1*ATR(2)=0.2)
        # and closes back below at 2008 with a bearish body -> SHORT signal.
        highs = [2005, 2006, 2005, 2011, 2006]
        lows = [2003, 2004, 2003, 2004, 2004]
        opens = [2004, 2005, 2004, 2009, 2005]
        closes = [2004, 2005, 2004, 2008, 2005]
        h, l, o, c, v = _mk_arrays(highs, lows, closes, opens)
        levels = [LiquidityLevel(2010.0, "PDH", "ABOVE")]
        sigs = detect_sweep_at_candle(3, h, l, o, c, v, atr=2.0, levels=levels, cfg=BASE_CFG)
        assert len(sigs) == 1
        s = sigs[0]
        assert s.direction == "SHORT"
        assert s.sweep_extreme == 2011.0
        assert s.level_price == 2010.0

    def test_long_after_sweep_of_lows(self):
        highs = [2006, 2006, 2006, 2007, 2006]
        lows = [2002, 2003, 2002, 1997, 2003]
        opens = [2005, 2005, 2005, 1999, 2004]
        closes = [2005, 2005, 2005, 2003, 2005]
        h, l, o, c, v = _mk_arrays(highs, lows, closes, opens)
        levels = [LiquidityLevel(2000.0, "PDL", "BELOW")]
        sigs = detect_sweep_at_candle(3, h, l, o, c, v, atr=2.0, levels=levels, cfg=BASE_CFG)
        assert len(sigs) == 1
        assert sigs[0].direction == "LONG"
        assert sigs[0].sweep_extreme == 1997.0

    def test_no_signal_without_poke(self):
        # Never exceeds the level -> no sweep
        highs = [2005, 2006, 2007, 2008, 2006]
        lows = [2003, 2004, 2004, 2004, 2004]
        closes = [2004, 2005, 2006, 2007, 2005]
        h, l, o, c, v = _mk_arrays(highs, lows, closes)
        levels = [LiquidityLevel(2010.0, "PDH", "ABOVE")]
        sigs = detect_sweep_at_candle(4, h, l, o, c, v, atr=2.0, levels=levels, cfg=BASE_CFG)
        assert sigs == []

    def test_no_signal_if_still_beyond_level(self):
        # Poked above and STAYS above (breakout, not sweep) -> no signal
        highs = [2005, 2006, 2011, 2013, 2014]
        lows = [2003, 2004, 2006, 2010, 2011]
        closes = [2004, 2005, 2011, 2012, 2013]
        h, l, o, c, v = _mk_arrays(highs, lows, closes)
        levels = [LiquidityLevel(2010.0, "PDH", "ABOVE")]
        sigs = detect_sweep_at_candle(4, h, l, o, c, v, atr=2.0, levels=levels, cfg=BASE_CFG)
        assert sigs == []

    def test_fires_once_not_on_following_bars(self):
        # Trigger at bar 3; bar 4 (still back inside, poke still in window) must NOT re-fire
        highs = [2005, 2006, 2005, 2011, 2006, 2005]
        lows = [2003, 2004, 2003, 2004, 2004, 2003]
        opens = [2004, 2005, 2004, 2009, 2005, 2004]
        closes = [2004, 2005, 2004, 2008, 2005, 2004]
        h, l, o, c, v = _mk_arrays(highs, lows, closes, opens)
        levels = [LiquidityLevel(2010.0, "PDH", "ABOVE")]
        assert len(detect_sweep_at_candle(3, h, l, o, c, v, 2.0, levels, BASE_CFG)) == 1
        assert detect_sweep_at_candle(4, h, l, o, c, v, 2.0, levels, BASE_CFG) == []

    def test_min_poke_respected(self):
        # Poke of only 0.1 with ATR 2.0 (needs >= 0.2) -> no signal
        highs = [2005, 2006, 2005, 2010.1, 2006]
        lows = [2003, 2004, 2003, 2004, 2004]
        opens = [2004, 2005, 2004, 2009, 2005]
        closes = [2004, 2005, 2004, 2008, 2005]
        h, l, o, c, v = _mk_arrays(highs, lows, closes, opens)
        levels = [LiquidityLevel(2010.0, "PDH", "ABOVE")]
        assert detect_sweep_at_candle(3, h, l, o, c, v, 2.0, levels, BASE_CFG) == []

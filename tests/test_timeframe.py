"""Tests for the run-timeframe layer: M5 -> M15/M30 resampling + config scaling."""

import pandas as pd
import pytest

from src.data.resample import resample_ohlcv, TIMEFRAME_MINUTES


def _m5_frame(start="2025-01-06 09:00", periods=12):
    """Sequential M5 candles with distinct values per bar."""
    ts = pd.date_range(start=start, periods=periods, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0 + i for i in range(periods)],
        "high": [110.0 + i for i in range(periods)],
        "low": [90.0 + i for i in range(periods)],
        "close": [105.0 + i for i in range(periods)],
        "volume": [10.0] * periods,
    })


class TestResample:
    def test_m5_passthrough(self):
        df = _m5_frame()
        out = resample_ohlcv(df, "M5")
        assert len(out) == len(df)
        assert (out["close"] == df["close"]).all()

    def test_m15_aggregation(self):
        df = _m5_frame(periods=6)  # two M15 bars
        out = resample_ohlcv(df, "M15")
        assert len(out) == 2
        # First M15 bar aggregates M5 bars 0-2
        assert out.iloc[0]["open"] == 100.0
        assert out.iloc[0]["high"] == 112.0
        assert out.iloc[0]["low"] == 90.0
        assert out.iloc[0]["close"] == 107.0
        assert out.iloc[0]["volume"] == 30.0

    def test_open_time_labels(self):
        df = _m5_frame(start="2025-01-06 09:00", periods=6)
        out = resample_ohlcv(df, "M15")
        assert out.iloc[0]["timestamp"] == pd.Timestamp("2025-01-06 09:00")
        assert out.iloc[1]["timestamp"] == pd.Timestamp("2025-01-06 09:15")

    def test_gaps_dropped_not_filled(self):
        df = pd.concat([
            _m5_frame(start="2025-01-06 09:00", periods=3),
            _m5_frame(start="2025-01-06 12:00", periods=3),
        ], ignore_index=True)
        out = resample_ohlcv(df, "M15")
        # No forward-filled bars between 09:15 and 12:00
        assert len(out) == 2
        assert out.iloc[1]["timestamp"] == pd.Timestamp("2025-01-06 12:00")

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError):
            resample_ohlcv(_m5_frame(), "M1")

    def test_misaligned_start_buckets_correctly(self):
        # Start at 09:05 — first M15 bucket [09:00, 09:15) holds only 2 bars
        df = _m5_frame(start="2025-01-06 09:05", periods=5)
        out = resample_ohlcv(df, "M15")
        assert out.iloc[0]["timestamp"] == pd.Timestamp("2025-01-06 09:00")
        assert out.iloc[0]["volume"] == 20.0
        assert out.iloc[1]["volume"] == 30.0


class TestApplyRunTimeframe:
    @pytest.fixture(autouse=True)
    def _restore_config(self):
        import config
        saved = {
            "halt": config.DRAWDOWN_CONTROLS["halt_max_bars"],
            "pre": config.NEWS_FILTER["pre_minutes"],
            "post": config.NEWS_FILTER["post_minutes"],
            "noentry": config.SESSION_FILTER["no_new_entries_before_rollover_minutes"],
            "closeroll": config.SESSION_FILTER["close_before_rollover_minutes"],
            "cooldown": config.SESSION_FILTER["cooldown_minutes_after_trade"],
            "tf": dict(config.RUN_TIMEFRAME),
        }
        yield
        config.DRAWDOWN_CONTROLS["halt_max_bars"] = saved["halt"]
        config.NEWS_FILTER["pre_minutes"] = saved["pre"]
        config.NEWS_FILTER["post_minutes"] = saved["post"]
        config.SESSION_FILTER["no_new_entries_before_rollover_minutes"] = saved["noentry"]
        config.SESSION_FILTER["close_before_rollover_minutes"] = saved["closeroll"]
        config.SESSION_FILTER["cooldown_minutes_after_trade"] = saved["cooldown"]
        config.RUN_TIMEFRAME.update(saved["tf"])

    def test_m5_is_noop(self):
        import config
        before_halt = config.DRAWDOWN_CONTROLS["halt_max_bars"]
        before_pre = config.NEWS_FILTER["pre_minutes"]
        config.apply_run_timeframe("M5")
        assert config.DRAWDOWN_CONTROLS["halt_max_bars"] == before_halt
        assert config.NEWS_FILTER["pre_minutes"] == before_pre
        assert config.RUN_TIMEFRAME["bar_minutes"] == 5

    def test_m15_scales_halt_and_floors_windows(self):
        import config
        config.apply_run_timeframe("M15")
        assert config.RUN_TIMEFRAME["bar_minutes"] == 15
        # 1440 M5 bars (~5 days) -> 480 M15 bars
        assert config.DRAWDOWN_CONTROLS["halt_max_bars"] == 480
        # Sub-bar windows floored to one bar
        assert config.NEWS_FILTER["pre_minutes"] >= 15
        assert config.SESSION_FILTER["close_before_rollover_minutes"] >= 15

    def test_m30_scaling(self):
        import config
        config.apply_run_timeframe("M30")
        assert config.DRAWDOWN_CONTROLS["halt_max_bars"] == 240
        assert config.NEWS_FILTER["pre_minutes"] >= 30

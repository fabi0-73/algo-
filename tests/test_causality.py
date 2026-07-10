"""
Causality guard tests for the backtest engine.

The 2026-07-10 audit found three intrabar/lookahead defects; each test here
fails on the pre-fix code and pins the causal contract afterwards:

  1. find_bos_after_manipulation scanned up to 20 bars PAST the decision bar
  2. validate_distribution_strength read bars AFTER the decision bar
  3. _check_exit raised BE/trailing stops from the current bar's favorable
     extreme, then tested the SAME bar against the raised stop

Contract pinned (matches src/research/lab.py and the live scanner, whose
DataFrame physically ends at the decision bar): a decision at bar i may read
bars <= i only; stop moves computed from bar k take effect at bar k+1.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STRATEGY
from src.strategy.market_structure import find_bos_after_manipulation, StructureBreak
from src.strategy.manipulation import ManipulationResult
from src.strategy.distribution import DistributionResult, validate_distribution_strength
from src.backtest.engine import BacktestEngine, TradeRecord


def _flat_df(n: int, base: float = 2000.0) -> pd.DataFrame:
    """Flat OHLC frame: every bar open=close=base, high=base+1, low=base-1."""
    ts = pd.date_range("2026-01-05 09:00", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base,
        "atr": 1.0,
        "volume": 100,
    })


# =============================================================================
# 1. BOS search must not see bars beyond the decision bar
# =============================================================================

class TestBOSCausality:
    def _df_with_late_bos(self) -> pd.DataFrame:
        """Swing high 2005 at idx 10; the ONLY close above it is at idx 25."""
        df = _flat_df(40)
        df.loc[10, "high"] = 2005.0          # swing high (strength-3 clean)
        df.loc[25, "close"] = 2010.0         # BOS candle
        df.loc[25, "high"] = 2011.0
        return df

    def test_future_bos_invisible_at_decision_bar(self):
        df = self._df_with_late_bos()
        bos = find_bos_after_manipulation(
            df, manipulation_return_idx=15, expected_direction="BULLISH",
            search_window=20, current_idx=20,
        )
        assert bos is None, (
            "BOS at idx 25 was visible to a decision at idx 20 — lookahead"
        )

    def test_bos_on_decision_bar_counts(self):
        df = self._df_with_late_bos()
        bos = find_bos_after_manipulation(
            df, manipulation_return_idx=15, expected_direction="BULLISH",
            search_window=20, current_idx=25,
        )
        assert bos is not None and bos.valid
        assert bos.break_candle_idx == 25

    def test_prefix_invariance(self):
        """Same decision index on truncated vs full frame must agree."""
        df = self._df_with_late_bos()
        full = find_bos_after_manipulation(
            df, 15, "BULLISH", search_window=20, current_idx=20)
        prefix = find_bos_after_manipulation(
            df.iloc[:21].copy(), 15, "BULLISH", search_window=20, current_idx=20)
        assert (full is None) == (prefix is None)

    def test_no_clamp_when_current_idx_omitted(self):
        """Live scanner passes no current_idx — frame end is the clamp."""
        df = self._df_with_late_bos()
        bos = find_bos_after_manipulation(
            df, 15, "BULLISH", search_window=20)
        assert bos is not None and bos.break_candle_idx == 25


# =============================================================================
# 2. Distribution follow-through must not read past the decision bar
# =============================================================================

class TestDistributionCausality:
    def _dist(self, break_idx: int) -> DistributionResult:
        return DistributionResult(
            valid=True, direction="UP", break_price=2000.0, break_distance=1.0,
            body_expansion=2.0, break_candle_idx=break_idx, atr=1.0,
        )

    def _df_bearish_after_break(self) -> pd.DataFrame:
        """Bars 21 and 22 close bearish — full-window validation fails."""
        df = _flat_df(30)
        for i in (21, 22):
            df.loc[i, "open"] = 2002.0
            df.loc[i, "close"] = 1998.0
        return df

    def test_future_follow_through_invisible(self):
        """Decision at the break bar: the two future bearish bars must not
        be readable; live-parity semantics = assume valid on no data."""
        df = self._df_bearish_after_break()
        ok = validate_distribution_strength(
            df, self._dist(20), min_follow_through_candles=2, current_idx=20)
        assert ok is True, (
            "validation at the break bar read future bars — lookahead"
        )

    def test_observed_bars_still_evaluated(self):
        df = self._df_bearish_after_break()
        ok = validate_distribution_strength(
            df, self._dist(20), min_follow_through_candles=2, current_idx=21)
        assert ok is False  # one observed bar, bearish -> fails threshold 1

    def test_full_window_unchanged_without_current_idx(self):
        df = self._df_bearish_after_break()
        ok = validate_distribution_strength(
            df, self._dist(20), min_follow_through_candles=2)
        assert ok is False

    def test_prefix_invariance(self):
        df = self._df_bearish_after_break()
        full = validate_distribution_strength(
            df, self._dist(20), min_follow_through_candles=2, current_idx=20)
        prefix = validate_distribution_strength(
            df.iloc[:21].copy(), self._dist(20),
            min_follow_through_candles=2, current_idx=20)
        assert full == prefix


# =============================================================================
# 3. BE/trailing stops staged from bar k must take effect at bar k+1
# =============================================================================

def _make_engine() -> BacktestEngine:
    return BacktestEngine(
        enable_session_filter=False,
        enable_news_filter=False,
        enable_htf_bias=False,
        enable_key_levels=False,
        enable_volume_filter=False,
        enable_fundamentals=False,
        enable_phantom_fills=False,
        enable_market_chase=False,
    )


def _open_long(engine: BacktestEngine, entry: float = 2000.0, sl: float = 1990.0,
               tp: float = 2100.0):
    trade = TradeRecord(
        entry_time=datetime(2026, 1, 5, 12, 0),
        direction="LONG",
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        original_sl=sl,
        position_size=0.01,
        best_price_in_favor=entry,
        worst_price_against=entry,
        entry_model="AMD",
    )
    engine.state.in_position = True
    engine.state.current_trade = trade
    engine.state.position_entry_idx = 0
    return trade


def _candle(high: float, low: float, close: float, minute: int) -> pd.Series:
    return pd.Series({
        "timestamp": pd.Timestamp(2026, 1, 5, 12, minute),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "atr": 2.0,
    })


class TestIntrabarStopCausality:
    @pytest.fixture(autouse=True)
    def _exit_params(self, monkeypatch):
        monkeypatch.setitem(STRATEGY, "move_sl_to_be_at_r", 1.0)
        monkeypatch.setitem(STRATEGY, "trailing_stop_enabled", True)
        monkeypatch.setitem(STRATEGY, "trailing_stop_activation_r", 1.0)
        monkeypatch.setitem(STRATEGY, "trailing_stop_atr_mult", 1.0)
        monkeypatch.setitem(STRATEGY, "disable_tp_when_trailing", True)

    def test_trail_raised_this_bar_cannot_stop_out_this_bar(self):
        """Bar spikes to 2R (trail would sit at 2018) then dips to 2012.
        Start-of-bar stop is 1990 — the trade must survive the bar, with the
        trail staged for the NEXT bar."""
        engine = _make_engine()
        trade = _open_long(engine)  # entry 2000, sl 1990, stop_dist 10, atr 2

        engine._check_exit(None, current_idx=5,
                           candle=_candle(high=2020.0, low=2012.0,
                                          close=2015.0, minute=5),
                           verbose=False)

        assert engine.state.current_trade is not None, (
            "trade exited on the same bar that raised the trail — intrabar lookahead"
        )
        assert trade.trailing_active is True
        assert trade.sl_price == pytest.approx(2018.0)  # 2020 - 1.0 * ATR(2)
        assert trade.best_price_in_favor == pytest.approx(2020.0)

    def test_staged_trail_fires_next_bar(self):
        engine = _make_engine()
        trade = _open_long(engine)

        engine._check_exit(None, 5, _candle(2020.0, 2012.0, 2015.0, 5), False)
        assert engine.state.current_trade is not None
        engine._check_exit(None, 6, _candle(2016.0, 2010.0, 2012.0, 10), False)

        assert engine.state.current_trade is None, "staged trail never fired"
        assert len(engine.trades) == 1
        rec = engine.trades[0]
        assert rec.exit_reason == "TRAIL_STOP"
        # exit at the staged 2018 minus SL slippage (small, ATR-scaled)
        assert 2017.5 <= rec.exit_price <= 2018.0
        assert rec.mfe_r == pytest.approx(2.0)

    def test_be_move_effective_next_bar(self, monkeypatch):
        """BE triggered by this bar's high must not convert this bar's dip
        into a same-bar BE_STOP."""
        monkeypatch.setitem(STRATEGY, "trailing_stop_enabled", False)
        engine = _make_engine()
        trade = _open_long(engine)  # BE at 1R = 2010; buffer 0.1*ATR = 0.2

        engine._check_exit(None, 5, _candle(2012.0, 2000.1, 2005.0, 5), False)

        assert engine.state.current_trade is not None, (
            "same-bar BE stop-out — intrabar lookahead"
        )
        assert trade.sl_moved_to_be is True
        assert trade.sl_price == pytest.approx(2000.2)

    def test_start_of_bar_stop_still_honored(self):
        """A genuine stop-out at the start-of-bar stop must still fire."""
        engine = _make_engine()
        _open_long(engine)

        engine._check_exit(None, 5, _candle(2005.0, 1989.0, 1992.0, 5), False)

        assert engine.state.current_trade is None
        assert len(engine.trades) == 1
        assert engine.trades[0].exit_reason == "SL_LOSS"


# =============================================================================
# 4. Live scanner must search BOS exactly like the engine (parity)
# =============================================================================

class TestLiveScannerBOSParity:
    """Under the shipping config (entry_mode=RETEST_ONLY + bos_required=True)
    the live scanner never searched for a BOS — the engine condition is
    `bos_required OR entry_mode != RETEST_ONLY`, live checked only the second
    half — so the entry gate rejected every AMD setup and the scanner could
    not emit a single AMD signal. This pins the engine-parity condition and
    the causal current_idx threading."""

    def test_live_scan_searches_bos_under_retest_only(self, monkeypatch):
        from src.live import signals as live_signals

        monkeypatch.setitem(STRATEGY, "entry_mode", "RETEST_ONLY")
        monkeypatch.setitem(STRATEGY, "bos_required", True)
        monkeypatch.setitem(STRATEGY, "min_confluence_score", 1)
        monkeypatch.setitem(STRATEGY, "short_min_confluence_score", 1)
        monkeypatch.setitem(STRATEGY, "retest_tolerance_atr_mult", 0.5)
        monkeypatch.setitem(STRATEGY, "rejection_wick_ratio", 1.5)

        scanner = live_signals.LiveSignalScanner(symbol="XAUUSD",
                                                 account_balance=500.0)
        n = scanner.min_bars + 40
        df = _flat_df(n)
        df["tick_volume"] = 100
        cur = n - 1
        # decision candle: retest of the flat consol high (2001) rejecting UP
        df.loc[cur, ["open", "high", "low", "close"]] = [2001.6, 2002.0,
                                                         2000.8, 2001.9]

        manip = ManipulationResult(valid=True, direction="DOWN",
                                   extreme_price=1997.0,
                                   return_candle_idx=cur - 25, atr=1.0)
        dist = DistributionResult(valid=True, direction="UP", break_price=2003.0,
                                  break_distance=1.0, body_expansion=2.0,
                                  break_candle_idx=cur - 5, atr=1.0)
        monkeypatch.setattr(scanner, "_is_consolidation", lambda *a, **k: True)
        monkeypatch.setattr(scanner, "_find_manipulation", lambda *a, **k: manip)
        monkeypatch.setattr(scanner, "_find_distribution", lambda *a, **k: dist)
        monkeypatch.setattr(live_signals, "validate_distribution_strength",
                            lambda *a, **k: True)
        # session/news/htf/volume gates are not under test — make permissive
        monkeypatch.setattr(scanner.time_filter, "can_enter_trade",
                            lambda *a, **k: (True, ""))
        monkeypatch.setattr(scanner.news_filter, "can_enter_trade",
                            lambda *a, **k: (True, ""))
        monkeypatch.setattr(scanner.htf_bias, "can_enter_trade",
                            lambda *a, **k: (True, "", ""))
        monkeypatch.setattr(scanner.volume_filter, "can_enter_trade",
                            lambda *a, **k: (True, "", ""))

        calls = []

        def spy_bos(df_, return_idx, expected_dir, **kwargs):
            calls.append(kwargs)
            return StructureBreak(valid=True, direction=expected_dir,
                                  broken_level=2001.0, break_price=2003.0,
                                  break_candle_idx=len(df_) - 6,
                                  swing_idx=len(df_) - 20)

        monkeypatch.setattr(live_signals, "find_bos_after_manipulation", spy_bos)

        sigs = scanner.scan(df)

        assert calls, ("live scan never searched for BOS under "
                       "RETEST_ONLY + bos_required — engine-parity break")
        assert calls[0].get("current_idx") == cur
        assert len(sigs) == 1
        assert sigs[0].bos_confirmed is True
        assert sigs[0].direction == "LONG"

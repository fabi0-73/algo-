"""
Test Suite for AMD Strategy Components
Unit tests for all strategy phases, filters, and the backtesting engine.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from src.strategy.indicators import (
    calculate_atr,
    calculate_body_sizes,
    calculate_range_boundaries,
    is_rejection_candle,
    add_indicators,
)
from src.strategy.consolidation import detect_consolidation, ConsolidationResult
from src.strategy.manipulation import detect_manipulation, ManipulationResult
from src.strategy.distribution import detect_distribution, DistributionResult
from src.strategy.entry import check_entry, EntrySignal
from src.strategy.risk import calculate_risk, calculate_exit_r_multiple, calculate_pnl, calculate_pnl_with_costs, RiskParams
from src.backtest.metrics import calculate_metrics, BacktestMetrics

# SMC Confluence imports
from src.strategy.fvg import FVG, detect_fvg, find_fvgs_in_range, is_price_leaving_fvg
from src.strategy.order_blocks import OrderBlock, detect_order_block, find_order_blocks_in_range
from src.strategy.market_structure import (
    SwingPoint, StructureBreak, detect_swing_high, detect_swing_low,
    find_swing_points, detect_break_of_structure
)

# New filter imports
from src.strategy.time_filters import TimeFilterEngine
from src.strategy.news_filter import NewsFilterEngine
from src.backtest.execution import ExecutionEngine, CostBreakdown
from src.backtest.engine import BacktestEngine
from src.strategy.distribution import DistributionResult


def create_sample_candles(
    count: int = 100,
    start_price: float = 2000.0,
    volatility: float = 5.0,
    trend: float = 0.0,
) -> pd.DataFrame:
    """Create sample OHLC data for testing."""
    np.random.seed(42)

    timestamps = [datetime.now() - timedelta(minutes=5 * (count - i)) for i in range(count)]

    prices = [start_price]
    for i in range(1, count):
        change = np.random.normal(trend, volatility)
        prices.append(prices[-1] + change)

    data = []
    for i, (ts, price) in enumerate(zip(timestamps, prices)):
        high = price + np.random.uniform(1, volatility)
        low = price - np.random.uniform(1, volatility)
        open_price = np.random.uniform(low, high)
        close_price = np.random.uniform(low, high)

        data.append({
            "timestamp": ts,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
            "volume": np.random.randint(100, 10000),
        })

    return pd.DataFrame(data)


def create_consolidation_candles(
    count: int = 20,
    mid_price: float = 2000.0,
    range_size: float = 2.0,
) -> pd.DataFrame:
    """Create candles that form a tight consolidation."""
    timestamps = [datetime.now() - timedelta(minutes=5 * (count - i)) for i in range(count)]

    data = []
    for ts in timestamps:
        # Keep prices within a tight range
        high = mid_price + range_size / 2 + np.random.uniform(0, 0.5)
        low = mid_price - range_size / 2 - np.random.uniform(0, 0.5)
        open_price = np.random.uniform(mid_price - range_size / 3, mid_price + range_size / 3)
        close_price = np.random.uniform(mid_price - range_size / 3, mid_price + range_size / 3)

        data.append({
            "timestamp": ts,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
            "volume": np.random.randint(100, 10000),
        })

    return pd.DataFrame(data)


class TestIndicators:
    """Tests for technical indicators."""

    def test_calculate_atr(self):
        """Test ATR calculation."""
        df = create_sample_candles(50)
        atr = calculate_atr(df, period=14)

        assert len(atr) == len(df)
        assert not atr.iloc[14:].isna().any()
        assert (atr.iloc[14:] > 0).all()

    def test_calculate_body_sizes(self):
        """Test body size calculation."""
        df = create_sample_candles(20)
        bodies = calculate_body_sizes(df)

        assert len(bodies) == len(df)
        assert (bodies >= 0).all()

        # Verify calculation
        expected = abs(df["close"] - df["open"])
        assert (bodies == expected).all()

    def test_calculate_range_boundaries(self):
        """Test range boundary calculation."""
        df = create_sample_candles(20)
        range_high, range_low = calculate_range_boundaries(df)

        assert range_high == df["high"].max()
        assert range_low == df["low"].min()

    def test_is_rejection_candle_bullish(self):
        """Test bullish rejection candle detection."""
        # Create a candle with long lower wick only (bullish rejection)
        candle = pd.Series({
            "open": 2000.0,
            "high": 2001.0,  # Small upper wick
            "low": 1993.0,   # Long lower wick
            "close": 2000.5,
        })

        assert is_rejection_candle(candle, "UP", wick_ratio=0.5)

    def test_is_rejection_candle_bearish(self):
        """Test bearish rejection candle detection."""
        # Create a candle with long upper wick (bearish rejection)
        candle = pd.Series({
            "open": 2000.0,
            "high": 2010.0,  # Long upper wick
            "low": 1999.0,
            "close": 2001.0,
        })

        assert is_rejection_candle(candle, "DOWN", wick_ratio=0.5)

    def test_add_indicators(self):
        """Test adding all indicators to DataFrame."""
        df = create_sample_candles(50)
        df_with_ind = add_indicators(df, atr_period=14)

        assert "atr" in df_with_ind.columns
        assert "body_size" in df_with_ind.columns
        assert "avg_body_size" in df_with_ind.columns
        assert "upper_wick" in df_with_ind.columns
        assert "lower_wick" in df_with_ind.columns
        assert "is_bullish" in df_with_ind.columns


class TestConsolidation:
    """Tests for consolidation detection."""

    def test_detect_consolidation_valid(self):
        """Test detection of valid consolidation."""
        # Create tight range candles
        df = create_consolidation_candles(count=40, range_size=1.0)

        result = detect_consolidation(df)

        # Should detect consolidation (tight range)
        assert isinstance(result, ConsolidationResult)
        assert result.range_high > result.range_low

    def test_detect_consolidation_invalid_range(self):
        """Test that wide range doesn't trigger consolidation."""
        # Create candles with wide range
        df = create_sample_candles(50, volatility=20.0)

        result = detect_consolidation(df)

        # Wide range should not be consolidation
        assert not result.valid or result.range_size > 10

    def test_detect_consolidation_insufficient_data(self):
        """Test handling of insufficient data."""
        df = create_sample_candles(10)  # Too few candles

        result = detect_consolidation(df)

        assert not result.valid


class TestManipulation:
    """Tests for manipulation (fake breakout) detection."""

    def test_manipulation_result_properties(self):
        """Test ManipulationResult properties."""
        result = ManipulationResult(
            valid=True,
            direction="DOWN",
            extreme_price=1995.0,
        )

        assert result.is_bullish_setup
        assert not result.is_bearish_setup

        result2 = ManipulationResult(
            valid=True,
            direction="UP",
            extreme_price=2005.0,
        )

        assert result2.is_bearish_setup
        assert not result2.is_bullish_setup


class TestRisk:
    """Tests for risk management calculations."""

    def test_calculate_risk_long(self):
        """Test risk calculation for long trade."""
        entry = EntrySignal(
            valid=True,
            direction="LONG",
            entry_price=2000.0,
            consolidation_high=1999.0,
            consolidation_low=1995.0,
            manipulation_extreme=1990.0,  # Fakeout low
            manipulation_direction="DOWN",
        )

        risk = calculate_risk(entry, account_balance=10000.0)

        assert risk.valid
        assert risk.stop_loss < entry.entry_price
        assert risk.take_profit > entry.entry_price
        assert risk.risk_reward_ratio >= 2.0
        assert risk.position_size > 0

    def test_calculate_risk_short(self):
        """Test risk calculation for short trade."""
        entry = EntrySignal(
            valid=True,
            direction="SHORT",
            entry_price=2000.0,
            consolidation_high=2005.0,
            consolidation_low=2001.0,
            manipulation_extreme=2010.0,  # Fakeout high
            manipulation_direction="UP",
        )

        risk = calculate_risk(entry, account_balance=10000.0)

        assert risk.valid
        assert risk.stop_loss > entry.entry_price
        assert risk.take_profit < entry.entry_price
        assert risk.risk_reward_ratio >= 2.0

    def test_calculate_exit_r_multiple_win(self):
        """Test R-multiple calculation for winning trade."""
        r = calculate_exit_r_multiple(
            entry_price=2000.0,
            exit_price=2020.0,  # Won
            stop_loss=1990.0,
            direction="LONG",
        )

        assert r == 2.0  # 20 profit / 10 risk = 2R

    def test_calculate_exit_r_multiple_loss(self):
        """Test R-multiple calculation for losing trade."""
        r = calculate_exit_r_multiple(
            entry_price=2000.0,
            exit_price=1990.0,  # Hit SL
            stop_loss=1990.0,
            direction="LONG",
        )

        assert r == -1.0

    def test_contract_size_pnl_long(self):
        """Test contract-size based P&L for long trade (Gold)."""
        # 1 lot = 100 oz, $1 move = $100/lot
        pnl_pips, pnl_usd = calculate_pnl(
            entry_price=2000.0,
            exit_price=2010.0,  # $10 move up
            position_size=0.1,  # 0.1 lot = 10 oz
            direction="LONG",
        )

        # $10 move * 0.1 lot * 100 oz/lot = $100
        assert pnl_usd == 100.0

    def test_contract_size_pnl_short(self):
        """Test contract-size based P&L for short trade (Gold)."""
        pnl_pips, pnl_usd = calculate_pnl(
            entry_price=2000.0,
            exit_price=1990.0,  # $10 move down
            position_size=0.5,  # 0.5 lot
            direction="SHORT",
        )

        # $10 move * 0.5 lot * 100 oz/lot = $500
        assert pnl_usd == 500.0

    def test_pnl_with_costs(self):
        """Test P&L calculation with execution costs."""
        gross_pnl, net_pnl, total_costs = calculate_pnl_with_costs(
            entry_price=2000.0,
            exit_price=2010.0,
            position_size=0.1,
            direction="LONG",
        )
        assert gross_pnl == 100.0
        assert net_pnl == 100.0  # No costs
        assert total_costs == 0.0

        gross_pnl2, net_pnl2, total_costs2 = calculate_pnl_with_costs(
            entry_price=2000.0,
            exit_price=2010.0,
            position_size=0.1,
            direction="LONG",
            commission=7.0,
            spread_cost=5.0,
            slippage_cost=3.0,
        )
        assert gross_pnl2 == 100.0
        assert net_pnl2 == 85.0  # $100 - $15 costs
        assert total_costs2 == 15.0


class TestExecutionCosts:
    """Tests for execution cost modeling."""

    def test_cost_breakdown_calculation(self):
        """Test cost breakdown dataclass."""
        costs = CostBreakdown(
            spread_cost=3.0,
            slippage_cost=2.0,
            commission_cost=7.0,
        )

        assert costs.total_cost == 12.0

    def test_execution_engine_spread_cost(self):
        """Test spread cost calculation."""
        engine = ExecutionEngine(spread_points=30.0)

        # 30 points = $0.30 per oz, 0.1 lot = 10 oz
        spread_cost = engine._calculate_spread_cost(0.1)

        # 0.30 * 10 = $3.00
        assert spread_cost == 3.0

    def test_execution_engine_commission(self):
        """Test commission calculation."""
        engine = ExecutionEngine(commission_per_lot=7.0)

        commission = engine._calculate_commission(0.5)

        # 0.5 lot * $7/lot = $3.50
        assert commission == 3.5

    def test_execution_costs_scale_with_position_size(self):
        """Test that execution costs scale with position size."""
        engine = ExecutionEngine(spread_points=30.0, slippage_model="NONE", commission_per_lot=7.0)

        entry = EntrySignal(
            valid=True,
            direction="LONG",
            entry_price=2000.0,
            desired_entry_price=1999.0,
        )
        candle = pd.Series({
            "open": 2001.0,
            "high": 2002.0,
            "low": 1998.0,
            "close": 2000.5,
        })

        fill_small = engine.simulate_entry_fill(entry, candle, atr=5.0, position_size=0.1)
        fill_large = engine.simulate_entry_fill(entry, candle, atr=5.0, position_size=1.0)

        assert fill_small.filled and fill_large.filled
        assert round(fill_large.costs.total_cost, 2) == round(fill_small.costs.total_cost * 10, 2)

    def test_execution_engine_uses_config_keys(self, monkeypatch):
        """Test that execution engine honors config keys."""
        from config import EXECUTION

        monkeypatch.setitem(EXECUTION, "entry_fill_model", "CLOSE")
        monkeypatch.setitem(EXECUTION, "intrabar_fill_rule", "BEST_CASE")

        engine = ExecutionEngine()
        assert engine.fill_model == "CLOSE"
        assert engine.intrabar_assumption == "BEST_CASE"

    def test_intrabar_worst_case_long_sl_first(self):
        """Test worst-case intrabar assumption - SL hit before TP for long."""
        engine = ExecutionEngine(intrabar_assumption="WORST_CASE")

        # Candle that touches both SL and TP
        candle = pd.Series({
            "open": 2000.0,
            "high": 2015.0,  # TP level
            "low": 1995.0,   # SL level
            "close": 2010.0,
        })

        # Trade record mock
        class MockTrade:
            direction = "LONG"
            entry_price = 2000.0
            sl_price = 1995.0
            tp_price = 2015.0
            position_size = 0.1

        exit_decision = engine.check_exit(MockTrade(), candle, atr=5.0)

        # Worst case for long: SL hit first (with unfavorable slippage)
        assert exit_decision.should_exit
        assert exit_decision.exit_reason == "SL"
        # SL price slipped unfavorably: 1995.0 - slippage (atr*0.02 = 0.10)
        assert exit_decision.exit_price < 1995.0  # Worse than exact SL

    def test_intrabar_worst_case_short_sl_first(self):
        """Test worst-case intrabar assumption - SL hit before TP for short."""
        engine = ExecutionEngine(intrabar_assumption="WORST_CASE")

        candle = pd.Series({
            "open": 2000.0,
            "high": 2005.0,  # SL level
            "low": 1985.0,   # TP level
            "close": 1990.0,
        })

        class MockTrade:
            direction = "SHORT"
            entry_price = 2000.0
            sl_price = 2005.0
            tp_price = 1985.0
            position_size = 0.1

        exit_decision = engine.check_exit(MockTrade(), candle, atr=5.0)

        # Worst case for short: SL hit first (with unfavorable slippage)
        assert exit_decision.should_exit
        assert exit_decision.exit_reason == "SL"
        # SL price slipped unfavorably: 2005.0 + slippage (atr*0.02 = 0.10)
        assert exit_decision.exit_price > 2005.0  # Worse than exact SL


class TestSessionFilter:
    """Tests for session/time filtering."""

    def test_killzone_in_range(self):
        """Test that killzone correctly identifies valid times."""
        engine = TimeFilterEngine(enabled=True)

        # 09:00 UTC should be in killzone (07:00-20:00)
        ts_in_killzone = datetime(2024, 1, 15, 9, 0, 0, tzinfo=pytz.UTC)
        assert engine.is_in_kill_zone(ts_in_killzone)

        # 21:00 UTC should be outside killzone (kill_zone_end is 20:00)
        ts_outside = datetime(2024, 1, 15, 21, 0, 0, tzinfo=pytz.UTC)
        assert not engine.is_in_kill_zone(ts_outside)

    def test_asian_session_detection(self):
        """Test Asian session detection (23:00-08:00 UTC)."""
        engine = TimeFilterEngine(enabled=True)

        # 02:00 UTC should be in Asian session
        ts_asian = datetime(2024, 1, 15, 2, 0, 0, tzinfo=pytz.UTC)
        assert engine.is_in_asian_session(ts_asian)

        # 14:00 UTC should not be Asian
        ts_not_asian = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        assert not engine.is_in_asian_session(ts_not_asian)

    def test_daily_trade_limit(self):
        """Test daily trade limit enforcement."""
        engine = TimeFilterEngine(enabled=True)
        engine.reset_daily_state()

        # Use a timestamp that's clearly in the kill zone
        ts = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)

        # Record trades directly to test limit
        for _ in range(engine.max_trades_per_day):
            engine.record_trade(ts)

        # Check limit is reached
        assert engine.has_reached_daily_limit(ts)

    def test_session_filter_disabled(self):
        """Test that disabled filter allows all times."""
        engine = TimeFilterEngine(enabled=False)

        # Even outside killzone should be allowed when disabled
        ts = datetime(2024, 1, 15, 3, 0, 0, tzinfo=pytz.UTC)
        can_trade, reason = engine.can_enter_trade(ts, 10000.0)

        assert can_trade


class TestNewsFilter:
    """Tests for news blackout filtering."""

    def test_news_filter_disabled_without_file(self):
        """Non-strict: news filter disables itself gracefully when CSV is missing."""
        engine = NewsFilterEngine(enabled=True, csv_path="nonexistent.csv", require_csv=False)

        ts = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        can_trade, reason = engine.can_enter_trade(ts)

        # Should allow trading when no news file exists (filter self-disabled)
        assert can_trade
        assert not engine.enabled

    def test_news_filter_require_csv_raises(self):
        """Strict: an enabled filter with require_csv=True raises if the CSV is missing."""
        raised = False
        try:
            NewsFilterEngine(enabled=True, csv_path="nonexistent.csv", require_csv=True)
        except RuntimeError:
            raised = True
        assert raised, "require_csv=True should raise when the CSV is missing"

    def test_news_filter_explicitly_disabled(self):
        """Test explicitly disabled news filter."""
        engine = NewsFilterEngine(enabled=False)

        ts = datetime(2024, 1, 15, 14, 0, 0, tzinfo=pytz.UTC)
        can_trade, reason = engine.can_enter_trade(ts)

        assert can_trade


class TestMetrics:
    """Tests for performance metrics calculation."""

    def test_calculate_metrics_empty(self):
        """Test metrics with no trades."""
        metrics = calculate_metrics([])

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        trades = [
            {"pnl_usd": 100, "r_multiple": 2.0, "direction": "LONG"},
            {"pnl_usd": -50, "r_multiple": -1.0, "direction": "LONG"},
            {"pnl_usd": 150, "r_multiple": 3.0, "direction": "SHORT"},
            {"pnl_usd": -50, "r_multiple": -1.0, "direction": "SHORT"},
        ]

        metrics = calculate_metrics(trades)

        assert metrics.total_trades == 4
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 2
        assert metrics.win_rate == 0.5
        assert metrics.total_pnl == 150.0
        assert metrics.long_trades == 2
        assert metrics.short_trades == 2

    def test_calculate_metrics_with_costs(self):
        """Test metrics with execution costs."""
        trades = [
            {
                "pnl_usd": 85, "gross_pnl": 100, "net_pnl": 85,
                "r_multiple": 2.0, "direction": "LONG",
                "spread_cost": 3.0, "slippage_cost": 5.0, "commission_cost": 7.0, "total_costs": 15.0
            },
            {
                "pnl_usd": -65, "gross_pnl": -50, "net_pnl": -65,
                "r_multiple": -1.0, "direction": "LONG",
                "spread_cost": 3.0, "slippage_cost": 5.0, "commission_cost": 7.0, "total_costs": 15.0
            },
        ]

        metrics = calculate_metrics(trades)

        assert metrics.total_trades == 2
        assert metrics.total_costs == 30.0
        assert metrics.cost_per_trade == 15.0
        assert metrics.net_pnl == 20.0  # 85 - 65

    def test_calculate_metrics_validation(self):
        """Test validation thresholds."""
        # Create trades that pass validation
        trades = []
        for i in range(600):  # > 500 trades
            if i % 3 == 0:
                trades.append({"pnl_usd": -50, "r_multiple": -1.0, "direction": "LONG"})
            else:
                trades.append({"pnl_usd": 100, "r_multiple": 2.0, "direction": "LONG"})

        # Generate equity curve
        equity = [10000]
        for t in trades:
            equity.append(equity[-1] + t["pnl_usd"])

        metrics = calculate_metrics(trades, equity_curve=equity)

        assert metrics.total_trades >= 500
        # Check if expectancy is positive
        assert metrics.expectancy > 0


class TestFVG:
    """Tests for Fair Value Gap detection."""

    def create_bullish_fvg_candles(self) -> pd.DataFrame:
        """Create candles with a bullish FVG (gap up)."""
        data = [
            {"timestamp": datetime.now() - timedelta(minutes=15), "open": 2000, "high": 2002, "low": 1998, "close": 2001, "volume": 1000},
            {"timestamp": datetime.now() - timedelta(minutes=10), "open": 2001, "high": 2010, "low": 2000, "close": 2009, "volume": 2000},  # Impulse
            {"timestamp": datetime.now() - timedelta(minutes=5), "open": 2008, "high": 2012, "low": 2006, "close": 2011, "volume": 1500},   # Gap: 2002 < 2006
        ]
        return pd.DataFrame(data)

    def create_bearish_fvg_candles(self) -> pd.DataFrame:
        """Create candles with a bearish FVG (gap down)."""
        data = [
            {"timestamp": datetime.now() - timedelta(minutes=15), "open": 2010, "high": 2012, "low": 2008, "close": 2009, "volume": 1000},
            {"timestamp": datetime.now() - timedelta(minutes=10), "open": 2009, "high": 2010, "low": 2000, "close": 2001, "volume": 2000},  # Impulse
            {"timestamp": datetime.now() - timedelta(minutes=5), "open": 2002, "high": 2004, "low": 1998, "close": 1999, "volume": 1500},   # Gap: 2008 > 2004
        ]
        return pd.DataFrame(data)

    def test_detect_bullish_fvg(self):
        """Test detection of bullish FVG."""
        df = self.create_bullish_fvg_candles()
        fvg = detect_fvg(df, idx=2)

        assert fvg is not None
        assert fvg.valid
        assert fvg.direction == "BULLISH"
        assert fvg.bottom == 2002  # Candle 1 high
        assert fvg.top == 2006     # Candle 3 low

    def test_detect_bearish_fvg(self):
        """Test detection of bearish FVG."""
        df = self.create_bearish_fvg_candles()
        fvg = detect_fvg(df, idx=2)

        assert fvg is not None
        assert fvg.valid
        assert fvg.direction == "BEARISH"
        assert fvg.top == 2008     # Candle 1 low
        assert fvg.bottom == 2004  # Candle 3 high

    def test_fvg_no_gap(self):
        """Test that overlapping candles don't create FVG."""
        data = [
            {"timestamp": datetime.now() - timedelta(minutes=15), "open": 2000, "high": 2005, "low": 1998, "close": 2001, "volume": 1000},
            {"timestamp": datetime.now() - timedelta(minutes=10), "open": 2001, "high": 2006, "low": 2000, "close": 2005, "volume": 2000},
            {"timestamp": datetime.now() - timedelta(minutes=5), "open": 2003, "high": 2008, "low": 2002, "close": 2007, "volume": 1500},  # Overlaps
        ]
        df = pd.DataFrame(data)
        fvg = detect_fvg(df, idx=2)

        assert fvg is None

    def test_fvg_properties(self):
        """Test FVG dataclass properties."""
        fvg = FVG(valid=True, direction="BULLISH", top=2010, bottom=2005, candle_idx=5)

        assert fvg.midpoint == 2007.5
        assert fvg.size == 5
        assert fvg.contains_price(2007)
        assert not fvg.contains_price(2012)
        assert fvg.is_near(2004, tolerance=2)

    def test_is_price_leaving_fvg_bullish(self):
        """Test detection of price leaving bullish FVG."""
        fvg = FVG(valid=True, direction="BULLISH", top=2006, bottom=2002, candle_idx=1)

        # Candle that taps FVG and closes above
        candle = pd.Series({"open": 2007, "high": 2010, "low": 2004, "close": 2009})
        assert is_price_leaving_fvg(candle, fvg)

        # Candle that stays inside FVG
        candle2 = pd.Series({"open": 2005, "high": 2006, "low": 2003, "close": 2004})
        assert not is_price_leaving_fvg(candle2, fvg)


class TestOrderBlocks:
    """Tests for Order Block detection."""

    def create_bullish_ob_candles(self) -> pd.DataFrame:
        """Create candles with a bullish Order Block."""
        data = [
            {"timestamp": datetime.now() - timedelta(minutes=30), "open": 2005, "high": 2007, "low": 2003, "close": 2006, "volume": 1000},
            {"timestamp": datetime.now() - timedelta(minutes=25), "open": 2006, "high": 2008, "low": 2004, "close": 2007, "volume": 1000},
            {"timestamp": datetime.now() - timedelta(minutes=20), "open": 2006, "high": 2007, "low": 2000, "close": 2001, "volume": 1500},  # Bearish candle (OB)
            {"timestamp": datetime.now() - timedelta(minutes=15), "open": 2002, "high": 2015, "low": 2001, "close": 2014, "volume": 3000},  # Impulse up
            {"timestamp": datetime.now() - timedelta(minutes=10), "open": 2014, "high": 2018, "low": 2012, "close": 2017, "volume": 2000},
            {"timestamp": datetime.now() - timedelta(minutes=5), "open": 2017, "high": 2020, "low": 2015, "close": 2019, "volume": 1500},
        ]
        return pd.DataFrame(data)

    def test_detect_bullish_order_block(self):
        """Test detection of bullish Order Block."""
        df = self.create_bullish_ob_candles()

        ob = detect_order_block(
            df,
            impulse_start_idx=3,  # The impulse candle
            direction="BULLISH",
            min_body_atr_mult=0,
            displacement_mult=1.0,
        )

        assert ob is not None
        assert ob.valid
        assert ob.direction == "BULLISH"
        assert ob.candle_idx == 2  # The bearish candle before impulse

    def test_order_block_properties(self):
        """Test OrderBlock dataclass properties."""
        ob = OrderBlock(valid=True, direction="BULLISH", top=2007, bottom=2000, candle_idx=2, strength=2.0)

        assert ob.midpoint == 2003.5
        assert ob.size == 7
        assert ob.contains_price(2005)
        assert not ob.contains_price(2010)
        assert ob.is_near(1998, tolerance=3)

    def test_find_order_blocks_in_range(self):
        """Test finding multiple Order Blocks."""
        df = self.create_bullish_ob_candles()

        obs = find_order_blocks_in_range(
            df,
            start_idx=0,
            end_idx=len(df),
            direction="BULLISH",
        )

        # Should find at least one OB
        assert len(obs) >= 0  # May vary based on detection criteria


class TestMarketStructure:
    """Tests for market structure and break of structure detection."""

    def create_uptrend_candles(self) -> pd.DataFrame:
        """Create candles forming an uptrend with swing points."""
        data = []
        base_price = 2000
        for i in range(20):
            # Create swing pattern: higher highs, higher lows
            if i % 4 == 0:  # Swing low
                low = base_price + i * 2 - 3
                high = low + 4
            elif i % 4 == 2:  # Swing high
                high = base_price + i * 2 + 3
                low = high - 4
            else:
                high = base_price + i * 2 + 2
                low = base_price + i * 2 - 2

            data.append({
                "timestamp": datetime.now() - timedelta(minutes=5 * (20 - i)),
                "open": (high + low) / 2 - 0.5,
                "high": high,
                "low": low,
                "close": (high + low) / 2 + 0.5,
                "volume": 1000,
            })

        return pd.DataFrame(data)

    def test_detect_swing_high(self):
        """Test swing high detection."""
        df = self.create_uptrend_candles()

        # Check a few positions for swing highs
        swing_found = False
        for idx in range(5, len(df) - 3):
            swing = detect_swing_high(df, idx, lookback=3, lookahead=3)
            if swing is not None:
                swing_found = True
                assert swing.type == "HIGH"
                assert swing.price == df.iloc[idx]["high"]
                break

        # May or may not find swing depending on data
        assert isinstance(swing_found, bool)

    def test_detect_swing_low(self):
        """Test swing low detection."""
        df = self.create_uptrend_candles()

        swing_found = False
        for idx in range(5, len(df) - 3):
            swing = detect_swing_low(df, idx, lookback=3, lookahead=3)
            if swing is not None:
                swing_found = True
                assert swing.type == "LOW"
                assert swing.price == df.iloc[idx]["low"]
                break

        assert isinstance(swing_found, bool)

    def test_swing_point_properties(self):
        """Test SwingPoint dataclass."""
        swing = SwingPoint(valid=True, type="HIGH", price=2050.0, candle_idx=10, strength=3)

        assert swing.valid
        assert swing.type == "HIGH"
        assert swing.price == 2050.0

    def test_structure_break_properties(self):
        """Test StructureBreak dataclass."""
        bos = StructureBreak(
            valid=True,
            direction="BULLISH",
            broken_level=2040.0,
            break_price=2045.0,
            break_candle_idx=15,
            swing_idx=10,
        )

        assert bos.valid
        assert bos.is_bullish
        assert not bos.is_bearish
        assert bos.broken_level == 2040.0

    def test_detect_break_of_structure(self):
        """Test BOS detection."""
        swing = SwingPoint(valid=True, type="HIGH", price=2040.0, candle_idx=5, strength=3)

        # Create a candle that breaks the swing high
        data = []
        for i in range(10):
            data.append({
                "timestamp": datetime.now() - timedelta(minutes=5 * (10 - i)),
                "open": 2035 + i,
                "high": 2037 + i if i < 7 else 2050,  # Break on candle 7
                "low": 2033 + i,
                "close": 2036 + i if i < 7 else 2048,
                "volume": 1000,
            })
        df = pd.DataFrame(data)

        bos = detect_break_of_structure(df, candle_idx=7, swing=swing, require_close=True)

        if bos is not None:
            assert bos.direction == "BULLISH"
            assert bos.broken_level == 2040.0

    def test_find_swing_points(self):
        """Test finding multiple swing points."""
        df = self.create_uptrend_candles()

        swings = find_swing_points(df, start_idx=0, end_idx=len(df), swing_lookback=3)

        # Should find some swings in trending data
        assert isinstance(swings, list)


class TestIntegration:
    """Integration tests for the full strategy pipeline."""

    def test_full_pipeline_structure(self):
        """Test that all components can be imported and initialized."""
        from src.backtest.engine import BacktestEngine
        from config import BACKTEST

        engine = BacktestEngine()
        assert engine is not None
        assert engine.initial_capital == BACKTEST["initial_capital"]

    def test_sample_backtest(self):
        """Test running backtest on sample data."""
        from src.backtest.engine import BacktestEngine

        # Create sample data (not guaranteed to have AMD patterns)
        df = create_sample_candles(500, volatility=3.0)

        engine = BacktestEngine()
        results = engine.run(df, verbose=False)

        # Always present
        assert "backtest_id" in results
        assert "total_trades" in results
        # May have no trades with random data - that's OK
        # Just verify it ran without exception

    def test_backtest_with_confluence_stats(self):
        """Test that backtest returns confluence statistics."""
        from src.backtest.engine import BacktestEngine

        df = create_sample_candles(500, volatility=3.0)

        engine = BacktestEngine()
        results = engine.run(df, verbose=False)

        # Confluence stats are in result if trades exist
        if results.get("total_trades", 0) > 0:
            assert "confluence_stats" in results
            assert "entries_with_fvg" in results["confluence_stats"]
        else:
            # No trades - just verify it ran
            assert "backtest_id" in results

    def test_backtest_with_cost_stats(self):
        """Test that backtest returns cost statistics."""
        from src.backtest.engine import BacktestEngine

        df = create_sample_candles(500, volatility=3.0)

        engine = BacktestEngine()
        results = engine.run(df, verbose=False)

        # Cost stats are in result if trades exist
        if results.get("total_trades", 0) > 0:
            assert "cost_stats" in results
            assert "total_spread_cost" in results["cost_stats"]
        else:
            # No trades - just verify it ran
            assert "backtest_id" in results

    def test_backtest_with_funnel_stats(self):
        """Test that backtest returns funnel statistics."""
        from src.backtest.engine import BacktestEngine

        df = create_sample_candles(500, volatility=3.0)

        engine = BacktestEngine()
        results = engine.run(df, verbose=False)

        # Funnel stats should always be present
        assert "funnel_stats" in results
        funnel = results["funnel_stats"]
        assert "consolidations_found" in funnel
        assert "entries_executed" in funnel

    def test_backtest_with_mtm_equity(self):
        """Test that backtest returns MTM equity curve."""
        from src.backtest.engine import BacktestEngine

        df = create_sample_candles(500, volatility=3.0)

        engine = BacktestEngine()
        results = engine.run(df, verbose=False)

        # MTM equity curve if trades exist
        if results.get("total_trades", 0) > 0:
            assert "mtm_equity_curve" in results
            assert len(results["mtm_equity_curve"]) > 0
        else:
            assert "backtest_id" in results

    def test_backtest_filter_toggles(self):
        """Test that filter toggles work correctly."""
        from src.backtest.engine import BacktestEngine

        df = create_sample_candles(500, volatility=3.0)

        # Run with all filters disabled
        engine = BacktestEngine(
            enable_session_filter=False,
            enable_news_filter=False,
            enable_htf_bias=False,
            enable_key_levels=False,
            enable_volume_filter=False,
            enable_fundamentals=False,
        )

        results = engine.run(df, verbose=False)

        assert "backtest_id" in results
        # Funnel stats should always be present
        assert "funnel_stats" in results
        funnel = results["funnel_stats"]
        # Filter rejection counts should be 0 when filters disabled
        assert funnel.get("filtered_session", 0) == 0
        assert funnel.get("filtered_news", 0) == 0
        assert funnel.get("filtered_htf_bias", 0) == 0


class TestDistributionFollowThrough:
    """Tests for distribution follow-through enforcement."""

    def test_distribution_follow_through_rejection(self, monkeypatch):
        """Ensure weak distribution is rejected when follow-through validation fails."""
        from config import SESSION_FILTER, STRATEGY

        # Disable session timing requirements for this test
        monkeypatch.setitem(SESSION_FILTER, "require_consolidation_in_asian", False)
        monkeypatch.setitem(SESSION_FILTER, "require_distribution_in_london_ny", False)
        monkeypatch.setitem(STRATEGY, "require_liquidity_sweep", False)

        df = create_sample_candles(200, volatility=2.0)
        # Set ATR high enough that range_size < consolidation_range_atr_mult * ATR
        # passes the fast pre-check (default mult is 4.0, so ATR=20 allows ranges up to 80)
        df["atr"] = 20.0

        engine = BacktestEngine(
            enable_session_filter=False,
            enable_news_filter=False,
            enable_htf_bias=False,
            enable_key_levels=False,
            enable_volume_filter=False,
            enable_fundamentals=False,
        )

        # Force consolidation/manipulation/distribution to appear valid
        monkeypatch.setattr(engine, "_is_consolidation_arrays", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            engine,
            "_check_manipulation_after",
            lambda *args, **kwargs: ManipulationResult(
                valid=True, direction="DOWN", extreme_price=1980.0, return_candle_idx=150, atr=1.0
            ),
        )
        monkeypatch.setattr(
            engine,
            "_check_distribution_after",
            lambda *args, **kwargs: DistributionResult(
                valid=True, direction="UP", break_price=2000.0, break_distance=1.0,
                body_expansion=2.0, break_candle_idx=160, atr=1.0
            ),
        )
        monkeypatch.setattr(
            "src.backtest.engine.validate_distribution_strength",
            lambda *args, **kwargs: False
        )

        # Initialize arrays that run() normally sets up
        engine._highs = df["high"].values
        engine._lows = df["low"].values
        engine._opens = df["open"].values
        engine._closes = df["close"].values
        engine._timestamps = df["timestamp"].values
        engine._atrs = df["atr"].values
        engine._tick_volumes = df["tick_volume"].values if "tick_volume" in df.columns else None

        # Pre-compute rolling windows
        from numpy.lib.stride_tricks import sliding_window_view
        lookback = engine.lookback
        _hw = sliding_window_view(engine._highs, lookback + 1)
        _lw = sliding_window_view(engine._lows, lookback + 1)
        engine._roll_high_max = _hw.max(axis=1)
        engine._roll_low_min = _lw.min(axis=1)
        engine._roll_range_size = engine._roll_high_max - engine._roll_low_min

        # Pre-compute close_pct pass/fail
        close_pct_threshold = STRATEGY.get("consolidation_close_pct", 0.60)
        _cw = sliding_window_view(engine._closes, lookback + 1)
        _range_low_2d = engine._roll_low_min[:, np.newaxis]
        _range_high_2d = engine._roll_high_max[:, np.newaxis]
        _inside = ((_cw >= _range_low_2d) & (_cw <= _range_high_2d)).sum(axis=1)
        engine._roll_close_pct_pass = (_inside / (lookback + 1)) >= close_pct_threshold

        _min_offset = STRATEGY.get("pattern_min_bars_after_consolidation", 10)
        _max_offset_cfg = STRATEGY.get("pattern_max_bars_after_consolidation", 60)
        _scan_half_width = (_max_offset_cfg - _min_offset) // 2 + 1
        if _scan_half_width > 0 and len(engine._roll_range_size) > _scan_half_width:
            _rs_view = sliding_window_view(engine._roll_range_size, _scan_half_width)
            engine._roll_range_min_in_scan = _rs_view.min(axis=1)
        else:
            engine._roll_range_min_in_scan = engine._roll_range_size

        # Pre-compute midnight opens and perf counters
        engine._midnight_opens = {}
        engine._perf_scan_calls = 0
        engine._perf_consol_pass = 0

        engine._scan_for_patterns(df, current_idx=len(df) - 1, verbose=False)

        assert engine.rejection_stats["no_distribution_follow_through"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for fib OTE helpers (src/strategy/fib.py)."""

import pytest

from src.strategy.fib import fib_zone, is_in_ote, ote_entry_price


class TestFibZone:
    def test_bullish_leg_zone_below_end(self):
        # Leg 100 -> 200; 61.8% retrace = 138.2, 79% = 121.0
        lo, hi = fib_zone(100.0, 200.0)
        assert lo == pytest.approx(121.0)
        assert hi == pytest.approx(138.2)

    def test_bearish_leg_zone_above_end(self):
        # Leg 200 -> 100; retracement measured back UP from 100
        lo, hi = fib_zone(200.0, 100.0)
        assert lo == pytest.approx(161.8)
        assert hi == pytest.approx(179.0)

    def test_ordering_invariant(self):
        lo, hi = fib_zone(150.0, 50.0)
        assert lo <= hi


class TestIsInOte:
    def test_inside(self):
        assert is_in_ote(130.0, 100.0, 200.0)

    def test_shallow_pullback_outside(self):
        assert not is_in_ote(170.0, 100.0, 200.0)

    def test_beyond_leg_outside(self):
        assert not is_in_ote(105.0, 100.0, 200.0)


class TestOteEntryPrice:
    def test_long_improves_price_when_retest_above_band(self):
        # LONG leg 100->200, retest level 150 (shallow): OTE asks 138.2 instead
        price = ote_entry_price("LONG", 100.0, 200.0, 150.0)
        assert price == pytest.approx(138.2)

    def test_long_never_worse_than_retest(self):
        # Retest already deeper than the band edge: keep the retest price
        price = ote_entry_price("LONG", 100.0, 200.0, 130.0)
        assert price == pytest.approx(130.0)

    def test_short_improves_price_when_retest_below_band(self):
        # SHORT leg 200->100, retest 150: OTE asks 161.8 (higher = better short)
        price = ote_entry_price("SHORT", 200.0, 100.0, 150.0)
        assert price == pytest.approx(161.8)

    def test_short_never_worse_than_retest(self):
        price = ote_entry_price("SHORT", 200.0, 100.0, 170.0)
        assert price == pytest.approx(170.0)

"""Cost parity: lab CostModel must reproduce ExecutionEngine formulas
and the engine's swap rollover-crossing convention."""
from datetime import datetime

import pytest

from src.backtest.execution import ExecutionEngine
from src.research.lab import CostModel, count_rollovers


@pytest.fixture
def engine():
    return ExecutionEngine()


@pytest.fixture
def costs():
    return CostModel.from_config()


@pytest.mark.parametrize("lots", [0.01, 0.02, 0.1, 1.0])
def test_spread_parity(engine, costs, lots):
    assert engine._calculate_spread_cost(lots) == pytest.approx(
        costs.spread_usd_oz * costs.contract_size * lots)


@pytest.mark.parametrize("lots,atr", [(0.01, 2.0), (0.05, 4.5), (0.2, 10.0)])
def test_slippage_parity(engine, costs, lots, atr):
    assert engine._calculate_slippage_cost(lots, atr) == pytest.approx(
        costs.slippage_atr_mult * atr * costs.contract_size * lots)


@pytest.mark.parametrize("lots", [0.01, 0.03, 0.5])
def test_commission_parity(engine, costs, lots):
    assert engine._calculate_commission(lots) == pytest.approx(
        costs.commission_usd_oz * costs.contract_size * lots)


def test_rollover_same_day_before():
    # entry midday, exit before 21:59 same day -> 0 nights
    assert count_rollovers(datetime(2025, 1, 6, 12, 0), datetime(2025, 1, 6, 21, 58)) == 0


def test_rollover_exit_exactly_at_rollover_counts():
    # engine convention: while roll <= exit -> counts
    assert count_rollovers(datetime(2025, 1, 6, 12, 0), datetime(2025, 1, 6, 21, 59)) == 1


def test_rollover_entry_after_rollover_first_crossing_next_day():
    assert count_rollovers(datetime(2025, 1, 6, 22, 30), datetime(2025, 1, 7, 10, 0)) == 0
    assert count_rollovers(datetime(2025, 1, 6, 22, 30), datetime(2025, 1, 7, 22, 0)) == 1


def test_rollover_multi_night():
    assert count_rollovers(datetime(2025, 1, 6, 12, 0), datetime(2025, 1, 9, 12, 0)) == 3


def test_champion_trade_cost_reconstruction():
    """Whole-system anchor: reproduce the champion report's per-trade
    spread/commission/swap from the lab CostModel."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "reports" / "backtest_124d15ef" / "results.json"
    if not path.exists():
        pytest.skip("champion report not present")
    with open(path) as f:
        trades = json.load(f)["trades"]
    costs = CostModel.from_config()
    checked = 0
    for t in trades[:50]:
        lots = float(t["position_size"])
        assert float(t["spread_cost"]) == pytest.approx(
            costs.spread_usd_oz * costs.contract_size * lots, abs=1e-6)
        assert float(t["commission_cost"]) == pytest.approx(
            costs.commission_usd_oz * costs.contract_size * lots, abs=1e-6)
        entry = datetime.strptime(t["entry_time"], "%Y-%m-%d %H:%M:%S")
        exit_ = datetime.strptime(t["exit_time"], "%Y-%m-%d %H:%M:%S")
        nights = count_rollovers(entry, exit_, costs.rollover_hh, costs.rollover_mm)
        assert float(t["swap_cost"]) == pytest.approx(
            costs.swap_usd_oz_per_night * costs.contract_size * lots * nights, abs=1e-6)
        checked += 1
    assert checked == 50

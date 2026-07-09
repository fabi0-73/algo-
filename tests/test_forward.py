"""Hand-computed forward returns/MFE/MAE, open[i+1] reference, NaN tails,
and cost arithmetic against config values."""
import numpy as np
import pandas as pd

from config import EXECUTION, RISK_MODEL
from src.research.forward import (
    baseline_pool, cost_in_atr, day_ids, forward_outcomes, net_r, tod_bucket,
)
from src.research.lab import CostModel


def tiny_frame():
    # bar:      0      1      2      3      4
    # open:   100    101    102    103    104
    # high:   100.5  101.8  102.6  103.9  104.2
    # low:     99.5  100.7  101.5  102.8  103.6
    # close:  101    102    103    104    104.1
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06 09:00", periods=5, freq="5min"),
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [100.5, 101.8, 102.6, 103.9, 104.2],
        "low": [99.5, 100.7, 101.5, 102.8, 103.6],
        "close": [101.0, 102.0, 103.0, 104.0, 104.1],
        "volume": 100,
    })
    df["atr"] = 2.0
    return df


def test_forward_returns_hand_computed():
    df = tiny_frame()
    fwd = forward_outcomes(df, horizons=(1, 3))
    # event at bar 0: ref = open[1] = 101, atr = 2
    assert np.isclose(fwd.loc[0, "fr_1"], (102.0 - 101.0) / 2.0)
    assert np.isclose(fwd.loc[0, "mfe_1"], (101.8 - 101.0) / 2.0)
    assert np.isclose(fwd.loc[0, "mae_1"], (100.7 - 101.0) / 2.0)
    # horizon 3 spans bars 1..3
    assert np.isclose(fwd.loc[0, "fr_3"], (104.0 - 101.0) / 2.0)
    assert np.isclose(fwd.loc[0, "mfe_3"], (103.9 - 101.0) / 2.0)
    assert np.isclose(fwd.loc[0, "mae_3"], (100.7 - 101.0) / 2.0)


def test_forward_nan_tails():
    df = tiny_frame()
    fwd = forward_outcomes(df, horizons=(1, 3))
    # fr_1 needs open[i+1] and close[i+1] -> last valid i = 3
    assert np.isnan(fwd.loc[4, "fr_1"])
    assert fwd["fr_1"].notna().sum() == 4
    # fr_3 needs close[i+3] -> last valid i = 1
    assert np.isnan(fwd.loc[2, "fr_3"]) and np.isnan(fwd.loc[4, "fr_3"])
    assert fwd["fr_3"].notna().sum() == 2


def test_cost_in_atr_matches_config():
    df = tiny_frame()  # atr = 2.0
    cost = CostModel.from_config()
    c = cost_in_atr(df, cost)
    spread = float(EXECUTION.get("fixed_spread_points", 30.0)) * 0.01
    contract = float(RISK_MODEL.get("contract_size", 100))
    commission = float(EXECUTION.get("commission_per_lot_round_turn", 7.0)) / contract
    expected = (spread + commission) / 2.0 + float(EXECUTION.get("slippage_atr_mult", 0.02))
    assert np.isclose(c.iloc[0], expected)
    # net_r subtracts it from the signed forward return
    fr = pd.Series([0.5] * 5)
    assert np.isclose(net_r(fr, c).iloc[0], 0.5 - expected)


def test_tod_bucket_and_day_ids():
    ts = pd.Series(pd.to_datetime([
        "2025-01-06 00:00", "2025-01-06 01:59", "2025-01-06 02:00",
        "2025-01-07 23:55",
    ]))
    b = tod_bucket(ts, bucket_minutes=120)
    assert b.tolist() == [0, 0, 1, 11]
    d = day_ids(ts)
    assert d.iloc[0] == d.iloc[1] == d.iloc[2]
    assert d.iloc[3] != d.iloc[0]


def test_baseline_pool_drops_nan():
    df = tiny_frame()
    fwd = forward_outcomes(df, horizons=(3,))
    pool = baseline_pool(fwd, tod_bucket(df["timestamp"]), 3)
    assert len(pool["values"]) == 2
    assert len(pool["buckets"]) == 2
    assert not np.isnan(pool["values"]).any()

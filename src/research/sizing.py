"""
$500-account equity replay with the real min-lot floor.

Uses floor() rounding (matches scripts/sizing_frontier.py, conservative;
the engine's risk.py rounds and can round UP — documented divergence).
Max-DD is MTM-approximated by injecting each trade's intra-trade trough
(from MAE) into the equity path; close-to-close DD understates badly.
"""
import numpy as np
import pandas as pd

from config import RISK_MODEL


def replay_equity(
    trades: pd.DataFrame,
    initial_capital: float = 500.0,
    risk_pct: float = 0.005,
    min_lot: float = None,
    lot_step: float = None,
    max_lot: float = None,
    contract_size: float = None,
) -> dict:
    min_lot = min_lot if min_lot is not None else float(RISK_MODEL.get("min_lot", 0.01))
    lot_step = lot_step if lot_step is not None else float(RISK_MODEL.get("lot_step", 0.01))
    max_lot = max_lot if max_lot is not None else float(RISK_MODEL.get("max_lot", 1.0))
    contract = contract_size if contract_size is not None else float(RISK_MODEL.get("contract_size", 100))

    if trades.empty:
        return {"final_equity": initial_capital, "max_dd_pct": 0.0, "max_dd_usd": 0.0,
                "win_rate": 0.0, "trades": 0, "monthly": {}, "profitable_months_pct": 0.0,
                "equity_curve": [initial_capital]}

    t = trades.sort_values("entry_time").reset_index(drop=True)
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    max_dd_usd = 0.0
    wins = 0
    monthly = {}
    curve = [equity]
    dir_sign = np.where(t["direction"].to_numpy() == "LONG", 1.0, -1.0)

    for i in range(len(t)):
        stop_dist = float(t["stop_dist"].iloc[i])
        risk_per_lot = stop_dist * contract
        lots = (equity * risk_pct) / risk_per_lot if risk_per_lot > 0 else min_lot
        lots = np.floor(lots / lot_step) * lot_step
        lots = min(max(lots, min_lot), max_lot)

        gross = dir_sign[i] * (float(t["exit"].iloc[i]) - float(t["entry"].iloc[i])) * contract * lots
        cost = float(t["cost_per_oz"].iloc[i]) * contract * lots
        net = gross - cost

        # intra-trade trough (MTM approximation)
        trough = equity - float(t["mae_r"].iloc[i]) * risk_per_lot * lots - cost
        for point in (trough, equity + net):
            if point > peak:
                peak = point
            dd = peak - point
            if peak > 0 and dd / peak > max_dd:
                max_dd = dd / peak
            if dd > max_dd_usd:
                max_dd_usd = dd

        equity += net
        curve.append(equity)
        if net > 0:
            wins += 1
        mkey = str(pd.Timestamp(t["entry_time"].iloc[i]).strftime("%Y-%m"))
        monthly[mkey] = monthly.get(mkey, 0.0) + net

    prof_months = sum(1 for v in monthly.values() if v > 0)
    return {
        "final_equity": float(equity),
        "max_dd_pct": float(max_dd),
        "max_dd_usd": float(max_dd_usd),
        "win_rate": wins / len(t),
        "trades": len(t),
        "monthly": monthly,
        "profitable_months_pct": prof_months / len(monthly) if monthly else 0.0,
        "equity_curve": curve,
    }

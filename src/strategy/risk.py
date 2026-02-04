"""
Phase 5: Risk Management
Calculate stop loss, take profit, and position sizing using contract-size math.

XAUUSD-correct implementation:
- 1 lot = 100 oz (contract_size = 100)
- $1 price move = $100 per lot
- No pip-based calculations (gold moves in $ points, not pips)
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from config import STRATEGY, BACKTEST, RISK_MODEL


@dataclass
class RiskParams:
    """Risk parameters for a trade."""
    valid: bool
    stop_loss: float = 0.0
    take_profit: float = 0.0
    stop_distance: float = 0.0  # Price distance to SL
    reward_distance: float = 0.0  # Price distance to TP
    risk_reward_ratio: float = 0.0
    position_size: float = 0.0  # In lots
    risk_amount_usd: float = 0.0  # USD at risk
    risk_per_lot_usd: float = 0.0  # Risk per 1 lot in USD
    notional_usd: float = 0.0  # Position notional value
    rejection_reason: str = ""

    # Legacy fields for compatibility
    risk_pips: float = 0.0
    reward_pips: float = 0.0
    risk_amount: float = 0.0  # Alias for risk_amount_usd

    def __post_init__(self):
        """Ensure risk_amount alias is set."""
        if self.risk_amount == 0.0 and self.risk_amount_usd > 0:
            object.__setattr__(self, 'risk_amount', self.risk_amount_usd)

    @property
    def r_multiple_if_win(self) -> float:
        """R-multiple if trade hits TP."""
        return self.risk_reward_ratio

    @property
    def r_multiple_if_loss(self) -> float:
        """R-multiple if trade hits SL (always -1)."""
        return -1.0


def calculate_risk(
    entry,  # EntrySignal - not typed to avoid circular import
    account_balance: float,
    min_rr: float = None,
    max_risk_pct: float = None,
    spread_buffer_pips: float = None,  # Legacy parameter, ignored if use_risk_model=True
    atr: float = None,
    use_risk_model: bool = True,
) -> RiskParams:
    """
    Calculate stop loss, take profit, and position size using contract-size math.

    Risk rules:
    1. SL = manipulation extreme + ATR buffer (beyond obvious level)
    2. SL must be at least min_stop_atr_mult * ATR from entry
    3. TP calculated for minimum risk:reward ratio
    4. Position size = risk_amount / (stop_distance * contract_size)
    5. Leverage guard: notional <= balance * max_notional_multiple

    Args:
        entry: Entry signal with price and context
        account_balance: Current account balance in USD
        min_rr: Minimum risk:reward ratio (default from config)
        max_risk_pct: Maximum risk per trade as decimal (default from RISK_MODEL)
        spread_buffer_pips: Legacy parameter (ignored if use_risk_model=True)
        atr: Current ATR value (for buffer calculations)
        use_risk_model: Use RISK_MODEL config (default True)

    Returns:
        RiskParams with calculated values
    """
    # Get config values
    if use_risk_model:
        contract_size = RISK_MODEL["contract_size"]
        risk_pct = max_risk_pct or RISK_MODEL["risk_pct_per_trade_default"]
        risk_pct = min(risk_pct, RISK_MODEL["risk_pct_per_trade_max"])
        min_stop_atr_mult = RISK_MODEL["min_stop_atr_mult"]
        stop_buffer_atr_mult = RISK_MODEL["stop_buffer_atr_mult"]
        max_notional_mult = RISK_MODEL["max_position_notional_multiple"]
        min_lot = RISK_MODEL["min_lot"]
        max_lot = RISK_MODEL["max_lot"]
        lot_step = RISK_MODEL["lot_step"]
    else:
        # Legacy fallback using pip-based calculations
        contract_size = BACKTEST.get("lot_size", 100)
        risk_pct = max_risk_pct or STRATEGY["max_risk_pct"]
        min_stop_atr_mult = 0.5
        stop_buffer_atr_mult = 0.0
        max_notional_mult = 10.0
        min_lot = 0.01
        max_lot = 100.0
        lot_step = 0.01

        # Use pip-based buffer if specified
        if spread_buffer_pips:
            pip_size = 0.1
            stop_buffer_price = spread_buffer_pips * pip_size
        else:
            stop_buffer_price = STRATEGY.get("spread_buffer_pips", 10) * 0.1

    min_rr = min_rr or STRATEGY["min_rr"]

    if not entry.valid:
        return RiskParams(valid=False, rejection_reason="invalid_entry")

    entry_price = entry.entry_price
    manipulation_extreme = entry.manipulation_extreme
    direction = entry.direction

    # Get ATR from entry or use provided
    if atr is None:
        # Try to get from entry context if available
        atr = getattr(entry, 'atr', None) or 1.0  # Default 1.0 to avoid division by zero

    # Calculate stop loss with buffer
    if use_risk_model:
        stop_buffer = stop_buffer_atr_mult * atr
        min_stop_distance = min_stop_atr_mult * atr
    else:
        stop_buffer = stop_buffer_price
        min_stop_distance = 0.5 * atr  # Minimum half ATR

    if direction == "LONG":
        # Long: SL below manipulation low
        raw_sl = manipulation_extreme - stop_buffer
        stop_distance = entry_price - raw_sl

        # Ensure minimum stop distance
        if stop_distance < min_stop_distance:
            stop_distance = min_stop_distance
            raw_sl = entry_price - stop_distance

        stop_loss = raw_sl

        # TP at minimum RR
        reward_distance = stop_distance * min_rr
        take_profit = entry_price + reward_distance

    else:  # SHORT
        # Short: SL above manipulation high
        raw_sl = manipulation_extreme + stop_buffer
        stop_distance = raw_sl - entry_price

        # Ensure minimum stop distance
        if stop_distance < min_stop_distance:
            stop_distance = min_stop_distance
            raw_sl = entry_price + stop_distance

        stop_loss = raw_sl

        # TP at minimum RR
        reward_distance = stop_distance * min_rr
        take_profit = entry_price - reward_distance

    # Validate stop distance
    if stop_distance <= 0:
        return RiskParams(valid=False, rejection_reason="invalid_stop_distance")

    # Calculate actual RR
    actual_rr = reward_distance / stop_distance

    if actual_rr < min_rr:
        return RiskParams(
            valid=False,
            rejection_reason=f"rr_too_low:{actual_rr:.2f}<{min_rr}"
        )

    # Calculate position size
    # risk_amount = account_balance * risk_pct
    # risk_per_lot = stop_distance * contract_size
    # lots = risk_amount / risk_per_lot

    risk_amount_usd = account_balance * risk_pct
    risk_per_lot_usd = stop_distance * contract_size
    lots = risk_amount_usd / risk_per_lot_usd

    # Round to lot step
    lots = round(lots / lot_step) * lot_step

    # Clamp to min/max
    lots = max(min_lot, min(lots, max_lot))

    # Leverage guard: check notional value
    notional = entry_price * contract_size * lots
    max_notional = account_balance * max_notional_mult

    if notional > max_notional:
        # Reduce lots to fit within leverage limit
        lots = max_notional / (entry_price * contract_size)
        lots = round(lots / lot_step) * lot_step
        lots = max(min_lot, lots)
        notional = entry_price * contract_size * lots

    # Final validation
    if lots < min_lot:
        return RiskParams(
            valid=False,
            rejection_reason=f"position_too_small:{lots}<{min_lot}"
        )

    # Legacy pip calculation for compatibility (1 pip = $0.10 for gold)
    pip_size = 0.1
    risk_pips = stop_distance / pip_size
    reward_pips = reward_distance / pip_size

    return RiskParams(
        valid=True,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        stop_distance=round(stop_distance, 2),
        reward_distance=round(reward_distance, 2),
        risk_reward_ratio=round(actual_rr, 2),
        position_size=lots,
        risk_amount_usd=round(risk_amount_usd, 2),
        risk_per_lot_usd=round(risk_per_lot_usd, 2),
        notional_usd=round(notional, 2),
        risk_pips=round(risk_pips, 1),
        reward_pips=round(reward_pips, 1),
        risk_amount=round(risk_amount_usd, 2),
    )


def calculate_trailing_stop(
    current_price: float,
    entry_price: float,
    current_sl: float,
    direction: str,
    trail_atr: float,
    trail_multiplier: float = 1.5,
) -> float:
    """
    Calculate trailing stop level.

    Args:
        current_price: Current market price
        entry_price: Original entry price
        current_sl: Current stop loss level
        direction: "LONG" or "SHORT"
        trail_atr: Current ATR value
        trail_multiplier: ATR multiplier for trailing distance

    Returns:
        New stop loss level (or original if not moved)
    """
    trail_distance = trail_atr * trail_multiplier

    if direction == "LONG":
        new_sl = current_price - trail_distance
        return max(new_sl, current_sl)  # Only move up
    else:
        new_sl = current_price + trail_distance
        return min(new_sl, current_sl)  # Only move down


def calculate_exit_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    direction: str,
) -> float:
    """
    Calculate R-multiple of a completed trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        stop_loss: Stop loss price
        direction: "LONG" or "SHORT"

    Returns:
        R-multiple (positive = profit, negative = loss)
    """
    if direction == "LONG":
        risk = entry_price - stop_loss
        if risk <= 0:
            return 0.0
        profit = exit_price - entry_price
    else:
        risk = stop_loss - entry_price
        if risk <= 0:
            return 0.0
        profit = entry_price - exit_price

    return round(profit / risk, 2)


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    position_size: float,
    direction: str,
    contract_size: float = None,
) -> Tuple[float, float]:
    """
    Calculate P&L in pips (legacy) and USD.

    Uses contract-size math:
    - gross_pnl_usd = (exit - entry) * contract_size * lots (for LONG)
    - gross_pnl_usd = (entry - exit) * contract_size * lots (for SHORT)

    Args:
        entry_price: Entry price
        exit_price: Exit price
        position_size: Position size in lots
        direction: "LONG" or "SHORT"
        contract_size: Contract size (default from RISK_MODEL)

    Returns:
        Tuple of (pnl_pips, pnl_usd)
    """
    if contract_size is None:
        contract_size = RISK_MODEL["contract_size"]

    # Legacy pip calculation
    pip_size = 0.1

    if direction == "LONG":
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    pnl_pips = price_diff / pip_size
    pnl_usd = price_diff * contract_size * position_size

    return round(pnl_pips, 1), round(pnl_usd, 2)


def calculate_pnl_with_costs(
    entry_price: float,
    exit_price: float,
    position_size: float,
    direction: str,
    commission: float = 0.0,
    spread_cost: float = 0.0,
    slippage_cost: float = 0.0,
    swap_cost: float = 0.0,
    contract_size: float = None,
) -> Tuple[float, float, float]:
    """
    Calculate gross and net P&L with cost breakdown.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        position_size: Position size in lots
        direction: "LONG" or "SHORT"
        commission: Commission cost in USD
        spread_cost: Spread cost in USD
        slippage_cost: Slippage cost in USD
        swap_cost: Swap cost in USD
        contract_size: Contract size (default from RISK_MODEL)

    Returns:
        Tuple of (gross_pnl_usd, net_pnl_usd, total_costs_usd)
    """
    if contract_size is None:
        contract_size = RISK_MODEL["contract_size"]

    if direction == "LONG":
        price_diff = exit_price - entry_price
    else:
        price_diff = entry_price - exit_price

    gross_pnl = price_diff * contract_size * position_size

    total_costs = commission + spread_cost + slippage_cost + swap_cost
    net_pnl = gross_pnl - total_costs

    return round(gross_pnl, 2), round(net_pnl, 2), round(total_costs, 2)


def validate_risk_params(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    direction: str,
    min_rr: float = None,
) -> Tuple[bool, str]:
    """
    Validate that risk parameters are logical.

    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        direction: "LONG" or "SHORT"
        min_rr: Minimum RR ratio

    Returns:
        Tuple of (valid, reason)
    """
    min_rr = min_rr or STRATEGY["min_rr"]

    if direction == "LONG":
        if stop_loss >= entry_price:
            return False, "sl_above_entry_for_long"
        if take_profit <= entry_price:
            return False, "tp_below_entry_for_long"

        risk = entry_price - stop_loss
        reward = take_profit - entry_price

    else:  # SHORT
        if stop_loss <= entry_price:
            return False, "sl_below_entry_for_short"
        if take_profit >= entry_price:
            return False, "tp_above_entry_for_short"

        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    if risk <= 0:
        return False, "zero_risk"

    rr = reward / risk
    if rr < min_rr:
        return False, f"rr_below_min:{rr:.2f}<{min_rr}"

    return True, ""

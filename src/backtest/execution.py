"""
Execution Model
Realistic trade execution with spread, slippage, commissions, and intrabar ambiguity.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum
import random
import pandas as pd
import numpy as np

from config import EXECUTION, RISK_MODEL


class IntrabarRule(Enum):
    """How to handle when both SL and TP are touched in same candle."""
    WORST_CASE = "WORST_CASE"
    BEST_CASE = "BEST_CASE"
    RANDOM = "RANDOM"


class FillModel(Enum):
    """Entry fill models."""
    CLOSE = "CLOSE"  # Legacy: fill at candle close
    LIMIT_AT_RETEST = "LIMIT_AT_RETEST"  # Fill at retest level if touched


@dataclass
class CostBreakdown:
    """Breakdown of all trading costs."""
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total costs in USD."""
        return self.spread_cost + self.slippage_cost + self.commission_cost


@dataclass
class FillResult:
    """Result of an entry fill attempt."""
    filled: bool
    fill_price: float = 0.0
    fill_model: str = ""
    fill_reason: str = ""
    costs: CostBreakdown = field(default_factory=CostBreakdown)


@dataclass
class ExitDecision:
    """Result of exit check for a position."""
    should_exit: bool
    exit_reason: str = ""
    exit_price: float = 0.0
    # Partial exit support
    is_partial: bool = False           # This is a partial close, not full exit
    partial_close_pct: float = 0.0     # Percentage to close
    new_sl_price: float = 0.0          # New SL after partial (for BE move)


class ExecutionEngine:
    """
    Realistic execution engine for backtesting.

    Handles:
    - Bid/ask spread modeling
    - Slippage (fixed or ATR-based)
    - Commission (round-turn)
    - Intrabar ambiguity (worst case default)
    - Limit order fill simulation
    """

    def __init__(
        self,
        fill_model: str = None,
        intrabar_assumption: str = None,
        spread_points: float = None,
        slippage_model: str = None,
        slippage_atr_mult: float = None,
        commission_per_lot: float = None,
        contract_size: float = None,
        random_seed: int = None,
    ):
        """
        Initialize execution engine.

        Args:
            fill_model: "CLOSE" or "LIMIT_AT_RETEST"
            intrabar_assumption: "WORST_CASE", "BEST_CASE", "RANDOM"
            spread_points: Spread in price points
            slippage_model: "NONE", "FIXED", "ATR_MULT"
            slippage_atr_mult: Slippage as ATR multiple
            commission_per_lot: Commission per lot (round-trip)
            contract_size: Contract size (oz per lot for XAU)
            random_seed: Seed for random decisions
        """
        self.fill_model = fill_model or EXECUTION.get("fill_model", "LIMIT_AT_RETEST")
        self.intrabar_assumption = intrabar_assumption or EXECUTION.get("intrabar_assumption", "WORST_CASE")
        self.spread_points = spread_points if spread_points is not None else EXECUTION.get("spread_points", 30.0)
        self.slippage_model = slippage_model or EXECUTION.get("slippage_model", "ATR_MULT")
        self.slippage_atr_mult = slippage_atr_mult if slippage_atr_mult is not None else EXECUTION.get("slippage_atr_mult", 0.1)
        self.commission_per_lot = commission_per_lot if commission_per_lot is not None else EXECUTION.get("commission_per_lot", 7.0)
        self.contract_size = contract_size if contract_size is not None else RISK_MODEL.get("contract_size", 100)

        seed = random_seed if random_seed is not None else EXECUTION.get("random_seed", 42)
        self.rng = random.Random(seed)

    def _calculate_spread_cost(self, position_size: float) -> float:
        """Calculate spread cost in USD."""
        # spread_points in price points, 1 point = $0.01 for XAUUSD
        # Cost = spread_points * 0.01 * contract_size * lots
        spread_usd_per_oz = self.spread_points * 0.01
        return spread_usd_per_oz * self.contract_size * position_size

    def _calculate_slippage(self, atr: float) -> float:
        """Calculate slippage in price points."""
        if self.slippage_model == "NONE":
            return 0.0
        elif self.slippage_model == "FIXED":
            return EXECUTION.get("slippage_points", 5.0)
        elif self.slippage_model == "ATR_MULT":
            return atr * self.slippage_atr_mult
        return 0.0

    def _calculate_slippage_cost(self, position_size: float, atr: float) -> float:
        """Calculate slippage cost in USD."""
        slippage_points = self._calculate_slippage(atr)
        return slippage_points * self.contract_size * position_size

    def _calculate_commission(self, position_size: float) -> float:
        """Calculate commission in USD."""
        return position_size * self.commission_per_lot

    def simulate_entry_fill(
        self,
        entry,  # EntrySignal object
        candle: pd.Series,
        atr: float,
    ) -> FillResult:
        """
        Simulate entry fill based on fill model.

        Args:
            entry: EntrySignal with desired entry details
            candle: Current candle OHLC
            atr: Current ATR value

        Returns:
            FillResult with fill status and costs
        """
        position_size = 0.1  # Default, will be overridden by risk calc

        if self.fill_model == "CLOSE":
            # Always fill at candle close (legacy mode)
            fill_price = candle["close"]
            filled = True
            fill_reason = "Filled at close"
        elif self.fill_model == "LIMIT_AT_RETEST":
            # Check if price retested the desired entry level
            desired_price = getattr(entry, 'desired_entry_price', None) or entry.entry_price

            if entry.direction == "LONG":
                # Long: need price to come down to our level
                filled = candle["low"] <= desired_price
                fill_price = desired_price if filled else 0.0
                fill_reason = "Limit filled at retest" if filled else "Limit not triggered"
            else:
                # Short: need price to come up to our level
                filled = candle["high"] >= desired_price
                fill_price = desired_price if filled else 0.0
                fill_reason = "Limit filled at retest" if filled else "Limit not triggered"
        else:
            # Default to close
            fill_price = candle["close"]
            filled = True
            fill_reason = "Filled at close (default)"

        if not filled:
            return FillResult(
                filled=False,
                fill_model=self.fill_model,
                fill_reason=fill_reason,
            )

        # Calculate costs
        spread_cost = self._calculate_spread_cost(position_size)
        slippage_cost = self._calculate_slippage_cost(position_size, atr)
        commission_cost = self._calculate_commission(position_size)

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_model=self.fill_model,
            fill_reason=fill_reason,
            costs=CostBreakdown(
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                commission_cost=commission_cost,
            ),
        )

    def check_exit(
        self,
        trade,  # TradeRecord object
        candle: pd.Series,
        atr: float,
    ) -> ExitDecision:
        """
        Check if position should be exited and at what price.

        Handles intrabar ambiguity when both SL and TP are touched.

        Args:
            trade: TradeRecord with position details
            candle: Current candle OHLC
            atr: Current ATR value

        Returns:
            ExitDecision with exit details
        """
        direction = trade.direction
        sl_price = trade.sl_price
        tp_price = trade.tp_price

        if direction == "LONG":
            sl_hit = candle["low"] <= sl_price
            tp_hit = candle["high"] >= tp_price
        else:
            sl_hit = candle["high"] >= sl_price
            tp_hit = candle["low"] <= tp_price

        if not sl_hit and not tp_hit:
            return ExitDecision(should_exit=False)

        # Handle intrabar ambiguity
        if sl_hit and tp_hit:
            if self.intrabar_assumption == "WORST_CASE":
                exit_at_sl = True
            elif self.intrabar_assumption == "BEST_CASE":
                exit_at_sl = False
            else:
                exit_at_sl = self.rng.random() < 0.5
        else:
            exit_at_sl = sl_hit

        if exit_at_sl:
            return ExitDecision(
                should_exit=True,
                exit_reason="SL",
                exit_price=sl_price,
            )
        else:
            return ExitDecision(
                should_exit=True,
                exit_reason="TP",
                exit_price=tp_price,
            )

    def check_exit_with_partial(
        self,
        trade,  # TradeRecord object
        candle: pd.Series,
        atr: float,
    ) -> ExitDecision:
        """
        Check for exit including partial TP logic.

        Order of checks:
        1. Full SL hit -> full exit
        2. Partial TP hit (if not taken) -> partial close + move SL to BE
        3. Final TP hit -> exit remainder

        Args:
            trade: TradeRecord with position details
            candle: Current candle OHLC
            atr: Current ATR value

        Returns:
            ExitDecision with exit or partial exit details
        """
        direction = trade.direction
        sl_price = trade.sl_price
        tp_price = trade.tp_price
        entry_price = trade.entry_price

        partial_enabled = RISK_MODEL.get("partial_tp_enabled", False)
        move_sl_to_be = RISK_MODEL.get("move_sl_to_be_after_partial", True)
        partial_pct = RISK_MODEL.get("partial_tp_at_1r", 0.5)

        # Calculate partial TP level (1R)
        # Use original_sl if available (SL before any BE move)
        original_sl = getattr(trade, 'original_sl', 0.0) or sl_price
        partial_tp_taken = getattr(trade, 'partial_tp_taken', False)

        if direction == "LONG":
            risk = entry_price - original_sl
            partial_tp = entry_price + risk  # 1R target

            sl_hit = candle["low"] <= sl_price
            partial_hit = not partial_tp_taken and candle["high"] >= partial_tp
            tp_hit = candle["high"] >= tp_price
        else:
            risk = original_sl - entry_price
            partial_tp = entry_price - risk  # 1R target

            sl_hit = candle["high"] >= sl_price
            partial_hit = not partial_tp_taken and candle["low"] <= partial_tp
            tp_hit = candle["low"] <= tp_price

        # Priority 1: SL hit = full exit
        if sl_hit:
            return ExitDecision(
                should_exit=True,
                exit_reason="SL",
                exit_price=sl_price,
                is_partial=False,
            )

        # Priority 2: Partial TP (if enabled and not taken)
        if partial_enabled and partial_hit:
            new_sl = entry_price if move_sl_to_be else sl_price
            return ExitDecision(
                should_exit=False,  # Not a full exit
                exit_reason="PARTIAL_TP",
                exit_price=partial_tp,
                is_partial=True,
                partial_close_pct=partial_pct,
                new_sl_price=new_sl,
            )

        # Priority 3: Final TP
        if tp_hit:
            return ExitDecision(
                should_exit=True,
                exit_reason="TP",
                exit_price=tp_price,
                is_partial=False,
            )

        return ExitDecision(should_exit=False)

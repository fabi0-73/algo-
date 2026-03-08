"""
Live Trading Monitor

Kill switch and risk management for live trading:
- Daily loss limit
- Account drawdown halt
- Regime filter (choppy market detection)
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MonitorState:
    """Tracks live trading session state."""
    trading_day: date = None
    daily_pnl: float = 0.0
    trades_today: int = 0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    choppy_streak: int = 0


class LiveMonitor:
    """
    Monitors live trading risk and enforces kill switches.

    Kill switch rules:
    1. Daily loss > daily_loss_limit_pct -> stop for the day
    2. Account drawdown > max_account_dd_pct -> halt until manual review
    3. Consecutive choppy regime checks >= choppy_halt_count -> pause
    """

    def __init__(
        self,
        initial_balance: float = 100.0,
        daily_loss_limit_pct: float = 0.01,
        max_account_dd_pct: float = 0.15,
        max_trades_per_day: int = 3,
        choppy_halt_count: int = 2,
    ):
        self.initial_balance = initial_balance
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_account_dd_pct = max_account_dd_pct
        self.max_trades_per_day = max_trades_per_day
        self.choppy_halt_count = choppy_halt_count

        self.state = MonitorState(
            current_balance=initial_balance,
            peak_balance=initial_balance,
        )

    def update_balance(self, new_balance: float) -> None:
        """Update account balance and check drawdown limits."""
        self.state.current_balance = new_balance
        if new_balance > self.state.peak_balance:
            self.state.peak_balance = new_balance

    def record_trade_result(self, pnl: float, timestamp: datetime = None) -> None:
        """Record a completed trade and check daily limits."""
        ts = timestamp or datetime.utcnow()
        today = ts.date()

        if self.state.trading_day != today:
            self.state.trading_day = today
            self.state.daily_pnl = 0.0
            self.state.trades_today = 0
            if self.state.halted and "daily" in self.state.halt_reason:
                self.state.halted = False
                self.state.halt_reason = ""

        self.state.daily_pnl += pnl
        self.state.trades_today += 1
        self.state.current_balance += pnl

        if self.state.current_balance > self.state.peak_balance:
            self.state.peak_balance = self.state.current_balance

    def record_regime(self, regime: str) -> None:
        """
        Record a regime check result.

        Args:
            regime: "TRENDING", "CHOPPY", or "NEUTRAL"
        """
        if regime.upper() == "CHOPPY":
            self.state.choppy_streak += 1
        else:
            self.state.choppy_streak = 0

    def can_trade(self, timestamp: datetime = None) -> tuple:
        """
        Check if trading is allowed based on all kill switch rules.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        ts = timestamp or datetime.utcnow()
        today = ts.date()

        if self.state.trading_day != today:
            self.state.trading_day = today
            self.state.daily_pnl = 0.0
            self.state.trades_today = 0
            if self.state.halted and "daily" in self.state.halt_reason:
                self.state.halted = False
                self.state.halt_reason = ""

        if self.state.halted and "account_drawdown" in self.state.halt_reason:
            return False, f"HALTED: {self.state.halt_reason} (manual review required)"

        # 1. Daily loss limit
        daily_limit = self.initial_balance * self.daily_loss_limit_pct
        if self.state.daily_pnl < -daily_limit:
            self.state.halted = True
            self.state.halt_reason = "daily_loss_exceeded"
            logger.warning(
                f"KILL SWITCH: Daily loss ${abs(self.state.daily_pnl):.2f} "
                f"exceeds limit ${daily_limit:.2f}"
            )
            return False, "daily_loss_exceeded"

        # 2. Account drawdown
        if self.state.peak_balance > 0:
            dd = (self.state.peak_balance - self.state.current_balance) / self.state.peak_balance
            if dd >= self.max_account_dd_pct:
                self.state.halted = True
                self.state.halt_reason = "account_drawdown_exceeded"
                logger.warning(
                    f"KILL SWITCH: Account DD {dd * 100:.1f}% "
                    f"exceeds limit {self.max_account_dd_pct * 100:.0f}%"
                )
                return False, "account_drawdown_exceeded"

        # 3. Max trades per day
        if self.state.trades_today >= self.max_trades_per_day:
            return False, "max_trades_per_day"

        # 4. Choppy regime
        if self.state.choppy_streak >= self.choppy_halt_count:
            return False, f"choppy_regime_{self.state.choppy_streak}_consecutive"

        return True, ""

    def force_resume(self) -> None:
        """Manually resume trading after a halt (for account_drawdown_exceeded)."""
        self.state.halted = False
        self.state.halt_reason = ""
        logger.info("Monitor resumed by operator")

    def status_summary(self) -> dict:
        """Return current monitor status as a dictionary."""
        dd = 0.0
        if self.state.peak_balance > 0:
            dd = (self.state.peak_balance - self.state.current_balance) / self.state.peak_balance

        return {
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "current_balance": round(self.state.current_balance, 2),
            "peak_balance": round(self.state.peak_balance, 2),
            "account_drawdown_pct": round(dd * 100, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "trades_today": self.state.trades_today,
            "choppy_streak": self.state.choppy_streak,
        }

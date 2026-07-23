"""Strategy registry. Modules are imported lazily so partially-built
rosters still screen (missing modules are skipped with a warning)."""
import importlib
import logging

logger = logging.getLogger(__name__)

STRATEGY_MODULES = [
    "asian_breakout",
    "displacement_pd",
    "ny_ib",
    "noise_area",
    "intraday_momentum",
    "pullback_window",
    "zone_bounce",
    "vwap_reversion",
    "rsi2_trend",
    "london_sweep_reversal",
    "htf_trend_pullback",
    "m15_trend_pullback",
]


def load_registry() -> dict:
    reg = {}
    for name in STRATEGY_MODULES:
        try:
            mod = importlib.import_module(f"src.research.strategies.{name}")
            reg[mod.NAME] = mod
        except Exception as e:  # noqa: BLE001 - partial rosters are expected
            logger.warning("strategy %s not loaded: %s", name, e)
    return reg

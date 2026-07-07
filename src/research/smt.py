"""
SMT (Smart Money Technique) divergence detector for gold.

At the AMD manipulation (a liquidity sweep), compare gold's swept extreme to a
correlated asset's swing over the same window. Divergence = the correlated
asset FAILED to confirm gold's sweep, which ICT reads as a genuine reversal.

  LONG setup  (manip swept a low): SMT+ if the correlated asset made a HIGHER
              low than its own prior-swing low (positive-corr: silver/EURUSD)
              or a HIGHER high failure (inverse-corr: DXY).
  SHORT setup (manip swept a high): mirror.

No lookahead: the manipulation extreme precedes the entry by several bars
(manipulation -> distribution -> BOS -> retest), so reading the correlated
asset around the manipulation time uses only information available before entry.

VERDICT 2026-07-08: KILLED by bucket analysis (scripts/smt_bucket.py on the
BOTH-mode AMD trades, run 3f3ea9d1). SMT divergence at the sweep did NOT
improve trades — it marked the WORSE ones, on TWO independent correlated
assets:
  Silver: SMT-present  n=72  61.1% WR  +0.421R  vs  SMT-absent n=25  76.0% WR +0.738R
  EURUSD: SMT-present  n=67  64.2% WR  +0.406R  vs  SMT-absent n=25  64.0% WR +0.681R
Delta (present-absent) = -0.32R (silver) / -0.28R (EURUSD): the opposite of
ICT doctrine. Interesting inverse hint — trades where the correlated asset
CONFIRMED gold's sweep did better — but n=25 is too thin and post-hoc, so not
pursued. Data caveat: broker only retains ~15mo M5 for secondary symbols, so
coverage was 97/143 trades (Feb 2025+, includes all OOS). Gate FAILED; no
engine integration. DXY (the canonical pairing) was unavailable on this broker.
"""
import numpy as np
import pandas as pd

# (ref window start, ref window end, current window back, current window fwd)
# in bars, relative to the manipulation timestamp. Current window straddles
# the sweep; ref window is the prior swing it swept.
DEFAULTS = {"ref_start": 36, "ref_end": 8, "cur_back": 8, "cur_fwd": 3}


def _win(df, t, back_bars, fwd_bars, bar_min=5):
    lo = t - pd.Timedelta(minutes=back_bars * bar_min)
    hi = t + pd.Timedelta(minutes=fwd_bars * bar_min)
    return df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]


def smt_divergence(corr_df, manip_time, direction, corr_kind="positive",
                   params=None):
    """Return (smt_present: bool | None, detail: dict).

    None = insufficient correlated data around this manipulation (skip).
    corr_kind: "positive" (silver, EURUSD) or "inverse" (DXY).
    """
    p = {**DEFAULTS, **(params or {})}
    manip_time = pd.Timestamp(manip_time)
    ref = _win(corr_df, manip_time - pd.Timedelta(minutes=p["ref_end"] * 5),
               p["ref_start"] - p["ref_end"], 0)
    cur = _win(corr_df, manip_time, p["cur_back"], p["cur_fwd"])
    if len(ref) < 3 or len(cur) < 2:
        return None, {}

    long_setup = direction == "LONG"
    # For a positive-corr asset, gold's low should coincide with the asset's
    # low; divergence = asset made a HIGHER low. For inverse-corr, gold's low
    # should coincide with the asset's HIGH; divergence = asset made a LOWER
    # high. Flip the compared extreme accordingly.
    use_low = (corr_kind == "positive") == long_setup
    if use_low:
        ref_ext = float(ref["low"].min())
        cur_ext = float(cur["low"].min())
        # divergence when the current extreme did NOT break the reference
        smt = cur_ext > ref_ext
    else:
        ref_ext = float(ref["high"].max())
        cur_ext = float(cur["high"].max())
        smt = cur_ext < ref_ext
    return bool(smt), {"ref_ext": ref_ext, "cur_ext": cur_ext,
                       "compared": "low" if use_low else "high"}


def load_corr(name):
    """Load a cached correlated-asset frame by concept name (e.g. 'xagusd')."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / f"lab_{name}_cache.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

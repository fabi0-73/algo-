"""
Shared cell machinery for scripts/event_study.py and scripts/mine_patterns.py.

A "cell" is one hypothesis test: (event x direction x horizon), optionally
conditioned on a context value. Values are direction-signed forward returns
in ATR units (so +excess always means "the hypothesis makes money before
costs"); the pool is signed the same way so the permutation null is fair.

Discipline encoded here, not left to callers:
  - decluster at the tested horizon (overlapping forward windows out)
  - min_n gate AFTER declustering
  - net_r = signed return minus the per-event round-trip cost
"""
import numpy as np
import pandas as pd

from .stats import decluster, summarize_edge


def directional_indices(ev: pd.DataFrame, mask: np.ndarray) -> dict:
    """Positional event indices under `mask`, split by hypothesis direction.
    Returns {direction: ndarray}; direction 0 = drift (no prior)."""
    fired = ev["fired"].to_numpy() & mask
    dirs = ev["direction"].to_numpy()
    out = {}
    for d in (-1, 0, 1):
        idx = np.where(fired & (dirs == d))[0]
        if idx.size:
            out[d] = idx
    return out


def evaluate_cell(
    idx: np.ndarray,
    direction: int,
    horizon: int,
    fr: np.ndarray,
    cost_atr: np.ndarray,
    days: np.ndarray,
    buckets: np.ndarray,
    pool_values: np.ndarray,
    pool_buckets: np.ndarray,
    min_n: int,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict:
    """One cell. fr is the full fr_<horizon> column (numpy, NaN tails);
    idx are candidate event positions. Returns a stats dict; cells below
    min_n come back with skipped=True so callers can report coverage
    honestly instead of silently dropping them."""
    n_raw = int(idx.size)
    idx = idx[decluster(idx, min_gap=horizon)]
    vals = fr[idx]
    ok = ~np.isnan(vals)
    idx, vals = idx[ok], vals[ok]
    if idx.size < min_n:
        return {"n_raw": n_raw, "n": int(idx.size), "skipped": True}

    sign = direction if direction != 0 else 1
    vals = sign * vals
    res = summarize_edge(vals, days[idx], buckets[idx],
                         sign * pool_values, pool_buckets,
                         n_boot=n_boot, n_perm=n_perm, seed=seed)
    res["n_raw"] = n_raw
    res["skipped"] = False
    res["cost_mean"] = float(np.mean(cost_atr[idx]))
    res["net_r_mean"] = float(np.mean(vals - cost_atr[idx]))
    res["event_idx"] = idx  # callers may want session splits; strip before json
    return res


def sign_consistent(idx: np.ndarray, vals: np.ndarray, boundary_pos: int) -> bool:
    """Same-sign mean in both halves of the train window (split at the
    positional boundary). Cells that flip sign are regime artifacts."""
    first = vals[idx < boundary_pos]
    second = vals[idx >= boundary_pos]
    if first.size == 0 or second.size == 0:
        return False
    return np.sign(first.mean()) == np.sign(second.mean()) != 0

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

from .stats import decluster, expected_null_mean, summarize_edge


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
    effect_floor: tuple = None,
) -> dict:
    """One cell. fr is the full fr_<horizon> column (numpy, NaN tails);
    idx are candidate event positions. Returns a stats dict; cells below
    min_n come back with skipped=True so callers can report coverage
    honestly instead of silently dropping them.

    effect_floor=(min_net_r, cost_mult): candidate-mining fast path. The
    TOD-matched excess is deterministic (expected_null_mean), so cells that
    cannot clear the effect gates are rejected BEFORE the 4k resamples.
    Fast-rejected cells keep p_value=1.0 — they stay in the BH-FDR
    denominator, so surviving cells get exactly the same adjusted p as a
    full run (p=1 sorts last and never passes). Never set this in the
    event study: its job is the honest table, including negative edges."""
    n_raw = int(idx.size)
    idx = idx[decluster(idx, min_gap=horizon)]
    vals = fr[idx]
    ok = ~np.isnan(vals)
    idx, vals = idx[ok], vals[ok]
    if idx.size < min_n:
        return {"n_raw": n_raw, "n": int(idx.size), "skipped": True}

    sign = direction if direction != 0 else 1
    vals = sign * vals
    cost_mean = float(np.mean(cost_atr[idx]))
    net_mean = float(np.mean(vals - cost_atr[idx]))

    if effect_floor is not None:
        min_net_r, cost_mult = effect_floor
        excess_det = float(vals.mean() - expected_null_mean(
            buckets[idx], sign * pool_values, pool_buckets))
        if excess_det < cost_mult * cost_mean or net_mean < min_net_r:
            return {
                "n_raw": n_raw, "n": int(idx.size), "skipped": False,
                "fast_reject": True, "p_value": 1.0,
                "mean": float(vals.mean()), "median": float(np.median(vals)),
                "excess": excess_det, "cost_mean": cost_mean,
                "net_r_mean": net_mean,
                "win_rate": float((vals > 0).mean()),
                "event_idx": idx,
            }

    res = summarize_edge(vals, days[idx], buckets[idx],
                         sign * pool_values, pool_buckets,
                         n_boot=n_boot, n_perm=n_perm, seed=seed)
    res["n_raw"] = n_raw
    res["skipped"] = False
    res["fast_reject"] = False
    res["cost_mean"] = cost_mean
    res["net_r_mean"] = net_mean
    # Cost fragility at 1.5x / 2x the modeled round-trip (report-only):
    # most kills are "real drift, too thin after costs" — make that visible
    # in-table instead of after a promotion cycle.
    res["net_r_mean_c15"] = float(np.mean(vals - 1.5 * cost_atr[idx]))
    res["net_r_mean_c20"] = float(np.mean(vals - 2.0 * cost_atr[idx]))
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

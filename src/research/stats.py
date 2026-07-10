"""
Numpy-only inference tools for event studies on overlapping bar data.

Design constraints (why these estimators):
  - Forward returns at horizon h overlap for events closer than h bars, so
    i.i.d. bootstrap/t-tests understate variance. decluster() removes the
    overlap; block_bootstrap_ci() resamples whole trading DAYS so intraday
    autocorrelation stays inside blocks.
  - Gold M5 has strong time-of-day structure (session vol regimes), so the
    permutation null must draw from the same time-of-day composition as the
    events, never from the whole pool.
  - Many (event x direction x horizon) cells are tested at once; bh_fdr()
    controls the false-discovery rate across the full grid.

All randomness goes through np.random.default_rng(seed) — callers pass seeds
so runs are reproducible (same convention as scripts/monte_carlo.py).
"""
import math

import numpy as np


def decluster(event_idx: np.ndarray, min_gap: int) -> np.ndarray:
    """Keep the first event of every cluster: drop events within min_gap bars
    of the last KEPT event. Input must be sorted ascending; returns a mask
    aligned to event_idx."""
    event_idx = np.asarray(event_idx)
    keep = np.zeros(len(event_idx), dtype=bool)
    last = None
    for i, idx in enumerate(event_idx):
        if last is None or idx - last >= min_gap:
            keep[i] = True
            last = idx
    return keep


def block_bootstrap_ci(
    values: np.ndarray,
    block_ids: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile CI for the mean, resampling whole blocks (trading days)
    with replacement. block_ids labels each value's block; blocks keep their
    internal correlation intact."""
    values = np.asarray(values, dtype=float)
    block_ids = np.asarray(block_ids)
    if values.size == 0:
        return {"mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_blocks": 0}
    # Resampled-sample mean = sum(block sums) / sum(block counts) — lets the
    # whole bootstrap run as two vectorized gathers instead of concatenations.
    uniq, inv = np.unique(block_ids, return_inverse=True)
    block_sums = np.bincount(inv, weights=values)
    block_counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = block_sums[draws].sum(axis=1) / block_counts[draws].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"mean": float(values.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_blocks": int(len(uniq))}


def perm_pvalue_excess(
    event_values: np.ndarray,
    event_buckets: np.ndarray,
    pool_values: np.ndarray,
    pool_buckets: np.ndarray,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict:
    """Two-sided permutation p-value for 'events beat the baseline'.

    Null: event bars are exchangeable with pool bars from the SAME
    time-of-day bucket. Each permutation draws, per bucket, as many pool
    values (with replacement) as there are events in that bucket, and takes
    the overall mean. excess = observed mean - null mean.
    """
    event_values = np.asarray(event_values, dtype=float)
    event_buckets = np.asarray(event_buckets)
    pool_values = np.asarray(pool_values, dtype=float)
    pool_buckets = np.asarray(pool_buckets)
    if event_values.size == 0:
        return {"excess": np.nan, "p_value": np.nan, "null_mean": np.nan}

    rng = np.random.default_rng(seed)
    obs = event_values.mean()
    n = event_values.size

    parts = []  # (pool subset, count) per bucket present in events
    for b in np.unique(event_buckets):
        pool_b = pool_values[pool_buckets == b]
        if pool_b.size == 0:  # no baseline for this bucket; fall back to full pool
            pool_b = pool_values
        parts.append((pool_b, int((event_buckets == b).sum())))

    null_means = np.zeros(n_perm)
    for pool_b, count in parts:
        idx = rng.integers(0, pool_b.size, size=(n_perm, count))
        null_means += pool_b[idx].sum(axis=1)
    null_means /= n

    null_mean = null_means.mean()
    p = (1.0 + np.sum(np.abs(null_means - null_mean) >= abs(obs - null_mean))) / (n_perm + 1.0)
    # Empirical p is floored at 1/(n_perm+1), which BH-FDR over a large grid
    # can never accept. When the observed mean is beyond every permutation
    # draw, extend the tail with a normal approximation (null means are
    # averages -> CLT), Phipson-Smyth-style hybrid.
    if p <= 1.0 / (n_perm + 1.0) + 1e-12:
        sd = null_means.std()
        if sd > 0:
            z = abs(obs - null_mean) / sd
            p = min(p, math.erfc(z / math.sqrt(2.0)))
    return {"excess": float(obs - null_mean), "p_value": float(p),
            "null_mean": float(null_mean)}


def bh_fdr(pvals: np.ndarray, q: float = 0.10) -> dict:
    """Benjamini-Hochberg. Returns reject mask and monotone adjusted p-values
    (NaN p -> never rejected, adjusted stays NaN)."""
    pvals = np.asarray(pvals, dtype=float)
    m = int(np.sum(~np.isnan(pvals)))
    reject = np.zeros(pvals.shape, dtype=bool)
    adj = np.full(pvals.shape, np.nan)
    if m == 0:
        return {"reject": reject, "p_adj": adj, "n_tests": 0}

    valid = np.where(~np.isnan(pvals))[0]
    order = valid[np.argsort(pvals[valid])]
    ranked = pvals[order]
    ranks = np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adj[order] = adj_sorted

    passed = np.where(ranked <= ranks / m * q)[0]
    if passed.size:
        reject[order[: passed[-1] + 1]] = True
    return {"reject": reject, "p_adj": adj, "n_tests": m}


def expected_null_mean(
    event_buckets: np.ndarray,
    pool_values: np.ndarray,
    pool_buckets: np.ndarray,
) -> float:
    """Deterministic TOD-matched null mean: per-bucket pool means weighted by
    the event bucket composition. This is the exact value the permutation
    null means converge to — cheap enough to gate on before resampling."""
    event_buckets = np.asarray(event_buckets)
    pool_values = np.asarray(pool_values, dtype=float)
    pool_buckets = np.asarray(pool_buckets)
    if event_buckets.size == 0:
        return np.nan
    total = 0.0
    for b in np.unique(event_buckets):
        pool_b = pool_values[pool_buckets == b]
        if pool_b.size == 0:
            pool_b = pool_values
        total += pool_b.mean() * (event_buckets == b).sum()
    return float(total / event_buckets.size)


def summarize_edge(
    event_values: np.ndarray,
    event_day_ids: np.ndarray,
    event_buckets: np.ndarray,
    pool_values: np.ndarray,
    pool_buckets: np.ndarray,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict:
    """One cell of the edge table: sample size, location, day-block CI,
    TOD-matched permutation excess/p — and the same excess/p machinery on
    the WIN RATE (P(value>0) vs the TOD-matched baseline hit rate), since a
    high-WR edge can exist with an unremarkable mean. Values are forward
    returns in ATR units (or net R) for one (event x direction x horizon)."""
    event_values = np.asarray(event_values, dtype=float)
    ok = ~np.isnan(event_values)
    event_values = event_values[ok]
    event_day_ids = np.asarray(event_day_ids)[ok]
    event_buckets = np.asarray(event_buckets)[ok]

    out = {"n": int(event_values.size)}
    if event_values.size == 0:
        out.update({"mean": np.nan, "median": np.nan, "ci_lo": np.nan,
                    "ci_hi": np.nan, "excess": np.nan, "p_value": np.nan,
                    "win_rate": np.nan, "wr_excess": np.nan,
                    "wr_p_value": np.nan})
        return out

    boot = block_bootstrap_ci(event_values, event_day_ids, n_boot=n_boot, seed=seed)
    perm = perm_pvalue_excess(event_values, event_buckets, pool_values,
                              pool_buckets, n_perm=n_perm, seed=seed)
    # Win-rate leg: identical machinery on the >0 indicator (seed offset so
    # the two nulls are not draw-correlated).
    pool_ok = ~np.isnan(pool_values)
    wr_perm = perm_pvalue_excess(
        (event_values > 0).astype(float), event_buckets,
        (pool_values[pool_ok] > 0).astype(float), pool_buckets[pool_ok],
        n_perm=n_perm, seed=seed + 1)
    out.update({
        "mean": float(event_values.mean()),
        "median": float(np.median(event_values)),
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "excess": perm["excess"],
        "p_value": perm["p_value"],
        "win_rate": float((event_values > 0).mean()),
        "wr_excess": wr_perm["excess"],
        "wr_p_value": wr_perm["p_value"],
    })
    return out

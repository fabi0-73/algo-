"""Decluster spacing, day-block bootstrap coverage, TOD-matched permutation
calibration, and a hand-worked BH-FDR example."""
import numpy as np

from src.research.stats import (
    bh_fdr, block_bootstrap_ci, decluster, expected_null_mean,
    perm_pvalue_excess, summarize_edge,
)
from src.research.study import evaluate_cell


def test_decluster_spacing():
    idx = np.array([10, 12, 15, 40, 44, 100])
    keep = decluster(idx, min_gap=6)
    assert keep.tolist() == [True, False, False, True, False, True]
    kept = idx[keep]
    assert (np.diff(kept) >= 6).all()


def test_decluster_all_kept_when_sparse():
    idx = np.array([0, 50, 100])
    assert decluster(idx, min_gap=10).all()


def test_block_bootstrap_constant_collapses():
    values = np.full(30, 0.7)
    days = np.repeat(np.arange(5), 6)
    r = block_bootstrap_ci(values, days, n_boot=200, seed=1)
    assert np.isclose(r["mean"], 0.7)
    assert np.isclose(r["ci_lo"], 0.7) and np.isclose(r["ci_hi"], 0.7)
    assert r["n_blocks"] == 5


def test_block_bootstrap_coverage():
    # CI should contain the true mean (0) close to nominal 95% over repeats.
    rng = np.random.default_rng(7)
    hits = 0
    reps = 100
    for k in range(reps):
        days = np.repeat(np.arange(20), 10)
        day_effect = np.repeat(rng.normal(0, 0.5, 20), 10)  # within-day correlation
        values = day_effect + rng.normal(0, 1.0, 200)
        r = block_bootstrap_ci(values, days, n_boot=300, seed=k)
        if r["ci_lo"] <= 0.0 <= r["ci_hi"]:
            hits += 1
    assert 0.82 <= hits / reps <= 1.0


def test_perm_pvalue_null_is_calibrated():
    # Events drawn FROM the pool -> p should be large most of the time.
    rng = np.random.default_rng(3)
    pool = rng.normal(0, 1, 5000)
    pool_buckets = np.tile(np.arange(5), 1000)
    small_ps = 0
    reps = 40
    for k in range(reps):
        pick = rng.integers(0, 5000, size=200)
        r = perm_pvalue_excess(pool[pick], pool_buckets[pick], pool,
                               pool_buckets, n_perm=300, seed=k)
        if r["p_value"] < 0.05:
            small_ps += 1
    assert small_ps <= reps * 0.15  # ~5% expected under the null


def test_perm_pvalue_detects_planted_edge():
    rng = np.random.default_rng(5)
    pool = rng.normal(0, 1, 5000)
    pool_buckets = np.tile(np.arange(5), 1000)
    events = rng.normal(0.5, 1, 300)  # +0.5 planted edge
    buckets = np.tile(np.arange(5), 60)
    r = perm_pvalue_excess(events, buckets, pool, pool_buckets, n_perm=500, seed=0)
    assert r["p_value"] < 0.01
    assert 0.3 < r["excess"] < 0.7


def test_perm_pvalue_respects_tod_buckets():
    # Pool: bucket 0 mean 0, bucket 1 mean +1. Events all from bucket 1 with
    # mean +1 have NO edge vs their own bucket; naive all-pool comparison
    # would call it an edge.
    rng = np.random.default_rng(9)
    pool = np.concatenate([rng.normal(0, 0.3, 2000), rng.normal(1, 0.3, 2000)])
    pool_buckets = np.concatenate([np.zeros(2000, int), np.ones(2000, int)])
    events = rng.normal(1, 0.3, 200)
    r = perm_pvalue_excess(events, np.ones(200, int), pool, pool_buckets,
                           n_perm=400, seed=1)
    assert r["p_value"] > 0.05
    assert abs(r["excess"]) < 0.15


def test_bh_fdr_hand_worked():
    # m=5, q=0.10: thresholds are 0.02, 0.04, 0.06, 0.08, 0.10.
    # sorted p = [0.01, 0.03, 0.20, 0.30, 0.90] -> largest k passing is k=2
    # (0.03 <= 0.04), so the two smallest are rejected.
    p = np.array([0.30, 0.01, 0.90, 0.03, 0.20])
    r = bh_fdr(p, q=0.10)
    assert r["reject"].tolist() == [False, True, False, True, False]
    assert r["n_tests"] == 5
    # adjusted p: monotone, p_adj[1] = min(0.01*5/1, 0.03*5/2, ...) = 0.05
    assert np.isclose(r["p_adj"][1], 0.05)
    assert np.isclose(r["p_adj"][3], 0.075)


def test_bh_fdr_handles_nan():
    p = np.array([0.001, np.nan, 0.5])
    r = bh_fdr(p, q=0.10)
    assert r["n_tests"] == 2
    assert not r["reject"][1] and np.isnan(r["p_adj"][1])
    assert r["reject"][0]


def test_summarize_edge_empty_and_nan():
    r = summarize_edge(np.array([]), np.array([]), np.array([]),
                       np.array([0.0, 1.0]), np.array([0, 0]))
    assert r["n"] == 0 and np.isnan(r["p_value"])
    r2 = summarize_edge(np.array([np.nan, 0.5, 0.5]), np.array([1, 1, 2]),
                        np.array([0, 0, 0]), np.random.default_rng(0).normal(0, 1, 500),
                        np.zeros(500, int), n_boot=100, n_perm=100)
    assert r2["n"] == 2  # NaN dropped


def test_expected_null_mean_matches_permutation():
    rng = np.random.default_rng(11)
    pool = np.concatenate([rng.normal(0, 0.3, 2000), rng.normal(1, 0.3, 2000)])
    pool_buckets = np.concatenate([np.zeros(2000, int), np.ones(2000, int)])
    event_buckets = np.array([0] * 50 + [1] * 150)
    det = expected_null_mean(event_buckets, pool, pool_buckets)
    perm = perm_pvalue_excess(np.zeros(200), event_buckets, pool, pool_buckets,
                              n_perm=2000, seed=2)
    # perm null_mean converges to the deterministic composition-weighted mean
    assert abs(det - perm["null_mean"]) < 0.02
    assert abs(det - (0.25 * 0.0 + 0.75 * 1.0)) < 0.05


def test_summarize_edge_win_rate_leg():
    # Pool is a fair +1/-1 coin per bucket; events win 80% -> WR excess ~0.3.
    rng = np.random.default_rng(13)
    pool = rng.choice([-1.0, 1.0], size=4000)
    pool_buckets = np.tile(np.arange(4), 1000)
    events = rng.choice([-1.0, 1.0], size=300, p=[0.2, 0.8])
    buckets = np.tile(np.arange(4), 75)
    days = np.repeat(np.arange(30), 10)
    r = summarize_edge(events, days, buckets, pool, pool_buckets,
                       n_boot=200, n_perm=500, seed=3)
    assert 0.72 <= r["win_rate"] <= 0.88
    assert 0.2 < r["wr_excess"] < 0.4
    assert r["wr_p_value"] < 0.01


def test_bh_fdr_p1_padding_equivalent_to_true_p():
    # Fast-rejected cells enter FDR with p=1.0; survivors' adjusted p and
    # rejection must be identical to a full run where those cells had their
    # true (large) p-values.
    full = bh_fdr(np.array([0.001, 0.40, 0.60]), q=0.10)
    padded = bh_fdr(np.array([0.001, 1.0, 1.0]), q=0.10)
    assert full["n_tests"] == padded["n_tests"] == 3
    assert np.isclose(full["p_adj"][0], padded["p_adj"][0])
    assert full["reject"].tolist() == padded["reject"].tolist()


def _cell_fixture(edge: float, seed: int = 17):
    rng = np.random.default_rng(seed)
    n = 6000
    fr = rng.normal(0, 1, n)
    idx = np.arange(100, 5800, 20)  # spacing 20 > any horizon: no declustering
    fr[idx] += edge
    cost = np.full(n, 0.15)
    days = np.repeat(np.arange(n // 100), 100)
    buckets = np.tile(np.arange(12), n // 12 + 1)[:n]
    ok = np.ones(n, dtype=bool)
    pool_v, pool_b = fr[ok], buckets[ok]
    return idx, fr, cost, days, buckets, pool_v, pool_b


def test_evaluate_cell_fast_reject_hopeless_cell():
    idx, fr, cost, days, buckets, pv, pb = _cell_fixture(edge=0.0)
    r = evaluate_cell(idx, 1, 6, fr, cost, days, buckets, pv, pb,
                      min_n=100, n_boot=100, n_perm=100, seed=0,
                      effect_floor=(0.05, 1.5))
    assert r["fast_reject"] is True
    assert r["p_value"] == 1.0
    assert "net_r_mean" in r and "excess" in r and "cost_mean" in r


def test_evaluate_cell_planted_edge_survives_fast_path():
    idx, fr, cost, days, buckets, pv, pb = _cell_fixture(edge=1.0)
    r = evaluate_cell(idx, 1, 6, fr, cost, days, buckets, pv, pb,
                      min_n=100, n_boot=100, n_perm=200, seed=0,
                      effect_floor=(0.05, 1.5))
    assert r["fast_reject"] is False
    assert r["p_value"] < 0.05
    assert r["net_r_mean"] > 0.5
    assert r["net_r_mean_c20"] < r["net_r_mean_c15"] < r["net_r_mean"]
    assert 0.0 <= r["win_rate"] <= 1.0

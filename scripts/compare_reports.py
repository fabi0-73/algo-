"""Compare two backtest reports side by side."""
import json
import sys
from collections import defaultdict
from datetime import datetime


def load_report(path):
    with open(path) as f:
        return json.load(f)


def monthly_pnl(trades):
    m = defaultdict(float)
    for t in trades:
        month = t.get('exit_time', t.get('entry_time', ''))[:7]
        m[month] += t.get('net_pnl', t.get('pnl_usd', 0))
    return dict(sorted(m.items()))


def monthly_count(trades):
    m = defaultdict(int)
    for t in trades:
        month = t.get('entry_time', '')[:7]
        m[month] += 1
    return dict(sorted(m.items()))


def confluence_dist(trades):
    d = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        s = t.get('confluence_score', 0)
        d[s]['count'] += 1
        if t.get('r_multiple', 0) > 0:
            d[s]['wins'] += 1
        d[s]['pnl'] += t.get('net_pnl', 0)
    return dict(sorted(d.items()))


def weekday_stats(trades):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    d = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        wd = datetime.fromisoformat(t['entry_time']).weekday()
        d[days[wd]]['count'] += 1
        if t.get('r_multiple', 0) > 0:
            d[days[wd]]['wins'] += 1
        d[days[wd]]['pnl'] += t.get('net_pnl', 0)
    return d


def direction_stats(trades):
    d = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        dir = t.get('direction', 'UNKNOWN')
        d[dir]['count'] += 1
        if t.get('r_multiple', 0) > 0:
            d[dir]['wins'] += 1
        d[dir]['pnl'] += t.get('net_pnl', 0)
    return d


def exit_stats(trades):
    d = defaultdict(int)
    for t in trades:
        d[t.get('exit_reason', 'unknown')] += 1
    return dict(d)


def wr(wins, count):
    return (wins / count * 100) if count else 0


def main(baseline_id, optimized_id):
    b = load_report(f'reports/backtest_{baseline_id}/results.json')
    o = load_report(f'reports/backtest_{optimized_id}/results.json')

    print('=' * 70)
    print(f'FULL COMPARISON: {baseline_id} (baseline) vs {optimized_id} (optimized)')
    print('=' * 70)

    # Core metrics
    print('\n--- CORE METRICS ---')
    metrics = [
        ('Total Trades', 'total_trades'),
        ('Wins', 'wins'),
        ('Losses', 'losses'),
        ('Win Rate %', 'win_rate'),
        ('Avg R-Multiple', 'avg_r_multiple'),
        ('Avg Win R', 'avg_win_r'),
        ('Avg Loss R', 'avg_loss_r'),
        ('Expectancy R', 'expectancy_r'),
        ('Gross P&L', 'gross_pnl_usd'),
        ('Net P&L', 'net_pnl_usd'),
        ('Profit Factor', 'profit_factor'),
        ('Max DD %', 'max_drawdown_pct'),
    ]
    print(f"{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
    print('-' * 58)
    for name, key in metrics:
        bv = b.get(key, 0)
        ov = o.get(key, 0)
        if isinstance(bv, float):
            print(f"{name:<20} {bv:>12.2f} {ov:>12.2f} {ov - bv:>+12.2f}")
        else:
            print(f"{name:<20} {bv:>12} {ov:>12} {ov - bv:>+12}")

    # Confidence tiers
    print('\n--- CONFIDENCE TIER BREAKDOWN ---')
    print(f"{'Tier':<12} {'B.Cnt':>6} {'B.WR%':>7} {'B.PnL':>10}  {'O.Cnt':>6} {'O.WR%':>7} {'O.PnL':>10}")
    print('-' * 65)
    for tier in ['high', 'standard', 'medium', 'base']:
        bd = b.get('confidence_tier_stats', {}).get(tier, {'count': 0, 'win_rate': 0, 'pnl': 0})
        od = o.get('confidence_tier_stats', {}).get(tier, {'count': 0, 'win_rate': 0, 'pnl': 0})
        if bd['count'] == 0 and od['count'] == 0:
            continue
        print(f"{tier:<12} {bd['count']:>6} {bd.get('win_rate', 0):>6.1f}% ${bd['pnl']:>+9.2f}  {od['count']:>6} {od.get('win_rate', 0):>6.1f}% ${od['pnl']:>+9.2f}")

    # Confluence scores
    print('\n--- CONFLUENCE SCORE DISTRIBUTION ---')
    bc = confluence_dist(b['trades'])
    oc = confluence_dist(o['trades'])
    all_scores = sorted(set(list(bc.keys()) + list(oc.keys())))
    print(f"{'Score':<8} {'B.Cnt':>6} {'B.WR%':>7} {'B.PnL':>10}  {'O.Cnt':>6} {'O.WR%':>7} {'O.PnL':>10}")
    print('-' * 62)
    for s in all_scores:
        bd = bc.get(s, {'count': 0, 'wins': 0, 'pnl': 0})
        od = oc.get(s, {'count': 0, 'wins': 0, 'pnl': 0})
        print(f"{s:<8} {bd['count']:>6} {wr(bd['wins'], bd['count']):>6.1f}% ${bd['pnl']:>+9.2f}  {od['count']:>6} {wr(od['wins'], od['count']):>6.1f}% ${od['pnl']:>+9.2f}")

    # Monthly P&L
    print('\n--- MONTHLY P&L ---')
    bm = monthly_pnl(b['trades'])
    om = monthly_pnl(o['trades'])
    bmc = monthly_count(b['trades'])
    omc = monthly_count(o['trades'])
    all_months = sorted(set(list(bm.keys()) + list(om.keys())))
    print(f"{'Month':<10} {'B.Trds':>7} {'B.PnL':>10}  {'O.Trds':>7} {'O.PnL':>10} {'Delta':>10}")
    print('-' * 58)
    b_neg = o_neg = 0
    for m in all_months:
        bv = bm.get(m, 0)
        ov = om.get(m, 0)
        bt = bmc.get(m, 0)
        ot = omc.get(m, 0)
        if bv <= 0:
            b_neg += 1
        if ov <= 0:
            o_neg += 1
        flag = ''
        if bv <= 0 and ov > 0:
            flag = ' FIXED'
        elif bv > 0 and ov <= 0:
            flag = ' BROKE'
        elif ov <= 0:
            flag = ' NEG'
        print(f"{m:<10} {bt:>7} ${bv:>+9.2f}  {ot:>7} ${ov:>+9.2f} ${ov - bv:>+9.2f}{flag}")
    print(f"{'Neg months':<10} {b_neg:>7} {'':>11} {o_neg:>7}")

    # Weekday
    print('\n--- WEEKDAY BREAKDOWN ---')
    bw = weekday_stats(b['trades'])
    ow = weekday_stats(o['trades'])
    print(f"{'Day':<6} {'B.Cnt':>6} {'B.WR%':>7} {'B.PnL':>10}  {'O.Cnt':>6} {'O.WR%':>7} {'O.PnL':>10}")
    print('-' * 60)
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        bd = bw.get(day, {'count': 0, 'wins': 0, 'pnl': 0})
        od = ow.get(day, {'count': 0, 'wins': 0, 'pnl': 0})
        print(f"{day:<6} {bd['count']:>6} {wr(bd['wins'], bd['count']):>6.1f}% ${bd['pnl']:>+9.2f}  {od['count']:>6} {wr(od['wins'], od['count']):>6.1f}% ${od['pnl']:>+9.2f}")

    # Direction
    print('\n--- DIRECTION BREAKDOWN ---')
    bdir = direction_stats(b['trades'])
    odir = direction_stats(o['trades'])
    print(f"{'Dir':<8} {'B.Cnt':>6} {'B.WR%':>7} {'B.PnL':>10}  {'O.Cnt':>6} {'O.WR%':>7} {'O.PnL':>10}")
    print('-' * 60)
    for d in ['LONG', 'SHORT']:
        bd = bdir.get(d, {'count': 0, 'wins': 0, 'pnl': 0})
        od = odir.get(d, {'count': 0, 'wins': 0, 'pnl': 0})
        print(f"{d:<8} {bd['count']:>6} {wr(bd['wins'], bd['count']):>6.1f}% ${bd['pnl']:>+9.2f}  {od['count']:>6} {wr(od['wins'], od['count']):>6.1f}% ${od['pnl']:>+9.2f}")

    # Exit reasons
    print('\n--- EXIT REASONS ---')
    be = exit_stats(b['trades'])
    oe = exit_stats(o['trades'])
    all_exits = sorted(set(list(be.keys()) + list(oe.keys())))
    print(f"{'Reason':<15} {'Baseline':>10} {'Optimized':>10}")
    print('-' * 37)
    for e in all_exits:
        print(f"{e:<15} {be.get(e, 0):>10} {oe.get(e, 0):>10}")

    # Costs
    print('\n--- COST ANALYSIS ---')
    print(f"{'Cost':<20} {'Baseline':>12} {'Optimized':>12}")
    print('-' * 46)
    for key in ['total_spread_cost', 'total_slippage_cost', 'total_commission_cost', 'total_costs']:
        bv = b.get('cost_stats', {}).get(key, 0)
        ov = o.get('cost_stats', {}).get(key, 0)
        label = key.replace('total_', '').replace('_', ' ').title()
        print(f"{label:<20} ${bv:>11.2f} ${ov:>11.2f}")

    # Top wins/losses
    sorted_t = sorted(o['trades'], key=lambda t: t.get('net_pnl', 0))
    print('\n--- TOP 5 LOSSES (Optimized) ---')
    for t in sorted_t[:5]:
        tier = t.get('confidence_tier', '?')
        score = t.get('confluence_score', 0)
        print(f"  {t['entry_time'][:16]} {t['direction']:>5} R:{t.get('r_multiple', 0):>+5.2f}  ${t['net_pnl']:>+8.2f}  tier:{tier}  score:{score}")

    print('\n--- TOP 5 WINS (Optimized) ---')
    for t in sorted_t[-5:]:
        tier = t.get('confidence_tier', '?')
        score = t.get('confluence_score', 0)
        print(f"  {t['entry_time'][:16]} {t['direction']:>5} R:{t.get('r_multiple', 0):>+5.2f}  ${t['net_pnl']:>+8.2f}  tier:{tier}  score:{score}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Compare two backtest reports side by side")
    parser.add_argument("baseline", help="Baseline report ID (reports/backtest_<id>)")
    parser.add_argument("optimized", help="Optimized report ID to compare against baseline")
    args = parser.parse_args()
    main(args.baseline, args.optimized)

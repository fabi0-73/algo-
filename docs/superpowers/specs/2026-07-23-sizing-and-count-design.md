# Design: Fixed-Ratio ladder, count atoms, HTF throttle (approved 2026-07-23)

Goals: lower DD, more trades, hold WR, maximize profit at $500. Full evidence
base: session runs + 3-agent expert research (scratchpad dd_*.md).

## Approved items
1. **Fixed-Ratio lot ladder** (Jones method, Davey-MC-calibrated): min-lot floor
   rises +0.01 per $DELTA of banked profit (equity - initial). Not martingale.
   DELTA chosen by Monte Carlo: max growth s.t. P(maxDD>25%) acceptable.
   Config-gated in RISK_MODEL; full gauntlet before adoption.
2. **Count atoms, evidence-first**: (a) failed-sweep reversal (bounded S&R after
   SL beyond manipulation extreme; Davey's 567k-backtest winner family),
   (b) skip-N re-entry (re-arm limit once post-stop while structure valid).
   Research/lab validation BEFORE any engine integration. Scheduled next session
   (engine surgery deserves fresh context).
3. **HTF execution throttle** (armed, dormant): 0.5x size while trailing-30d
   stream P&L < 0, activates only when HTF execution unlocks at >=$1,250.
   Justification: HTF trade-R lag-1 autocorr +0.376 (the literature's required
   precondition); AMD's is ~0 so AMD gets NO throttle.

## Killed by evidence (do not revisit without new data)
Equity-throttle on AMD; affordability gate (wide-stop trades = 96% of profit);
pyramiding; time-stop recycling; concurrent multi-TF. Mid-stop-tercile skip is
a PRE-REGISTERED hypothesis for future-data validation only.

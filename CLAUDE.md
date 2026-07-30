# Forex Modeling Project — CLAUDE.md

## Purpose

Build a home research pipeline for systematic FX trading: acquire clean historical
data, engineer features, build/backtest statistical or ML models, validate
out-of-sample with proper walk-forward methodology, and (only after that)
paper-trade before any live capital is considered.

This is a research-first project. No live trading, no broker API integration,
until the backtest/validation stage is solid and reviewed.

---

## 0. CURRENT STATE — READ THIS FIRST

_Last updated 2026-07-19 (session 13). 200 tests passing._
_Repo: github.com/ZHCooke/Predicto-Forex_

### 0.0b RESULT: 21.5 years settles it — the effect is REAL, and cost-bound

EURUSD 2005-2026 complete: **143,701 mid bars, 0 errors, 21.5 years.**

**The London-open effect over the full history (gross pips, EURUSD):**

| era | n | pips | t | p |
|---|---|---|---|---|
| **2005-2014 (FRESH, never looked at)** | 2975 | **1.402** | **2.984** | **0.0029** |
| 2015-2022 (research — where we selected) | 2076 | 2.876 | 5.912 | <1e-6 |
| 2023-2026 (holdout) | 907 | 1.025 | 1.707 | 0.088 |
| **FULL 21.5 YEARS** | **5958** | **1.858** | **6.119** | **1e-9** |

**20 of 22 years positive.** Decay trend +0.024 pips/yr, p = 0.63 — no decay.

**Three conclusions, now firmly established:**

1. **The effect is real.** p = 1e-9 over 21.5 years, and independently
   confirmed at p = 0.0029 on 2005-2014 data that played no part in finding it.
   This is the first genuinely replicated result in the project.
2. **Its true size is ~1.3 pips, not 2.9.** Excluding the selection window
   entirely gives **1.314 pips** across 3,882 out-of-sample trades. The research
   period overstated by **2.2x** — a textbook demonstration of selection bias,
   and note the h002 holdout figure (1.025) was far closer to truth than the
   research figure that generated the prediction.
3. **It is not being arbitraged away.** No decay across 21 years, spanning the
   GFC, the rise of algo trading, and a 5x spread compression (EURUSD median
   went 1.00 pip in 2005 to 0.30 by 2014). The effect held its size while costs
   collapsed around it.

**IT IS A COST PROBLEM, NOT A SIGNAL PROBLEM. Breakeven cost is 1.31 pips:**

| cost scenario | round trip | net pips | Sharpe |
|---|---|---|---|
| retail now (our measured) | 0.80 | 0.514 | **0.337** |
| tight retail / ECN | 0.50 | 0.814 | **0.534** |
| institutional | 0.30 | 1.014 | 0.666 |
| prime + rebate | 0.15 | 1.164 | 0.764 |

At our measured 0.80 pip round trip it earns Sharpe 0.337 — real, but below the
s8 bar of 0.5. **At 0.50 pips it clears the bar.** That is the whole question
now, and it is a question about execution, not about models.

Note this also explains why the effect never disappeared: at 2005-2012 spreads
of 1.0-1.5 pips it was never profitable to arbitrage away, and by the time
spreads compressed it sat just under the retail cost line.

### 0.0c The cost assumption interrogated — it reduces to ONE unmeasured number

Full basket now complete: **all 7 pairs, 2005-2026, ~144k mid bars each, zero
validation errors** (NZDUSD starts 2007).

**The effect is a USD phenomenon, and it is coherent.** USD strengthens in the
London morning on **7 of 7 pairs**, significant on 3:

| pair | 21y gross | t | p | modern spread | round trip | net |
|---|---|---|---|---|---|---|
| **EURUSD** | **1.858** | **6.12** | <1e-6 | 0.30 | 0.50 | **+1.358** |
| USDJPY | 1.124 | 3.46 | 0.0006 | 0.40 | 0.60 | +0.524 |
| USDCHF | 1.143 | 2.67 | 0.008 | 1.00 | 1.20 | -0.057 |
| NZDUSD | 0.433 | 1.17 | 0.24 | 1.10 | 1.30 | -0.867 |
| GBPUSD | 0.427 | 1.16 | 0.24 | 0.80 | 1.00 | -0.573 |
| USDCAD | 0.105 | 0.36 | 0.72 | 1.10 | 1.30 | -1.195 |
| AUDUSD | 0.054 | 0.15 | 0.88 | 1.00 | 1.20 | -1.146 |

Strong in the European/funding currencies (EUR, JPY, CHF), absent in the
commodity dollars — economically coherent for a London-morning flow story.
(Caveat: the pairs share the USD leg, so 7/7 is not 7 independent confirmations.)

**A multi-pair book does NOT help. EURUSD alone is the best portfolio:**

| book | net pips | Sharpe | CI |
|---|---|---|---|
| **EURUSD alone** | 1.358 | **0.916** | [0.569, 1.283] |
| EURUSD + USDJPY | 0.983 | 0.808 | [0.430, 1.211] |
| all 7 equal-weight | -0.280 | -0.243 | [-0.603, 0.134] |

Diversification cannot rescue pairs whose spread exceeds their edge. Adding
USDJPY halves the edge while correlation (0.21) is too high to halve the risk.

**READ THE HEADLINE NUMBER CAREFULLY.** The 0.916 above spans all 21 years
INCLUDING the 2015-2022 window we selected on. The honest, selection-free
figure — 2005-2014 plus 2023-2026 only — is:

| round trip | net pips | Sharpe | 95% CI |
|---|---|---|---|
| 1.10 (spread + 0.40/leg slip) | 0.214 | 0.140 | [-0.271, 0.566] |
| 0.80 (our original assumption) | 0.514 | 0.337 | [-0.081, 0.765] |
| **0.50 (spread + 0.10/leg slip)** | **0.814** | **0.534** | **[0.113, 0.961]** |
| 0.30 (limit orders, spread only) | 1.014 | 0.666 | [0.241, 1.094] |

**Use 0.534, not 0.916.**

**THE WHOLE PROJECT NOW REDUCES TO ONE UNMEASURED NUMBER: SLIPPAGE.**

- The **spread is measured**, at the exact hours traded: 0.30 pips at both the
  08:00 entry and the 12:00 exit, stable since 2014.
- The **edge is measured**, selection-free, over 21 years: 1.314 pips.
- **Slippage is assumed and has never been validated.** It is the only input
  standing between Sharpe 0.34 and Sharpe 0.53, and it CANNOT be measured from
  bar data — it requires live execution records.

**One structural argument in its favour:** this is a CALENDAR rule with no
information urgency. At 07:59 London you already know you will trade at 08:00,
so the order can be worked passively rather than crossing the spread — unlike a
signal that must be executed the moment it fires. Passive execution in the most
liquid hour of the day plausibly means slippage near zero, or negative. But
"plausibly" is doing real work in that sentence, and it is exactly the kind of
assumption this project has repeatedly found to be wrong.

### 0.0d Tick-level execution study — the cost assumption was 2.3x too harsh

`src/analysis/execution.py`. Reconstructs the actual fill from real bid/ask
ticks at the decision instant, on a random sample of 30 days from 2024-2025
(~80k ticks/day).

**A BAR-LABELLING BUG FOUND HERE, AND IT MATTERS FOR THE TRADE SPEC.**
Dukascopy bars are **START-labelled**: a bar stamped 08:00 opens at 08:00 and
closes at 09:00. Verified against ticks to 0.1 pip on all three of 07/08/09.

Our analysis uses the CLOSE of the London-hour-8 bar, so the actual trade is
**enter 09:00 London, exit 13:00 London** — not 08:00-12:00 as the prose in
earlier sessions loosely said. The h002 pre-registration is unaffected because
it was written precisely ("that bar close"), and every statistical result is
unaffected because the bar analysis was internally consistent throughout. But
the first tick study ran an hour early on both legs and produced -5.6 pips,
which is what caught it. **Any live implementation must use 09:00 London.**

**MEASURED EXECUTION CONDITIONS** (corrected timing, 09:00 London entry):

| | median | p90 |
|---|---|---|
| spread at entry instant | **0.30 pips** | 0.61 |
| spread at exit instant | **0.30 pips** | 0.50 |
| ticks in a +/-60s window | 251 | — |

**Round-trip cost crossing the spread: 0.345 pips** — not the 0.80 we assumed
for eight sessions. The assumption was 2.3x too harsh.

The 30-day P&L (mean -4.5, median +1.8 to +2.3 pips) is NOT a test of the edge:
at 25-pip daily vol, 30 observations give t = -0.94, i.e. noise. The median
being +1.8/+2.3 is consistent with the historical median (2.41 research, 1.89
holdout), which is mildly reassuring, and that is all it is. The study's
purpose was to measure the SPREAD, which it did.

### 0.0e WHERE THIS LANDS — viable, with one narrow unknown left

Selection-free (2005-2014 + 2023-2026, n = 3,882), using tick-measured costs:

| scenario | round trip | net pips | Sharpe | 95% CI |
|---|---|---|---|---|
| old assumption (unvalidated) | 0.800 | 0.514 | 0.337 | [-0.081, 0.765] |
| **tick-measured, crossing spread** | **0.345** | **0.969** | **0.636** | **[0.212, 1.064]** |
| + 0.10/leg broker slippage | 0.545 | 0.769 | 0.505 | [0.084, 0.932] |
| + 0.20/leg broker slippage | 0.745 | 0.569 | 0.374 | [-0.044, 0.803] |
| passive fills (earns spread) | -0.345 | 1.659 | 1.089 | [0.663, 1.523] |

**At the measured cost this clears the s8 bar with a CI excluding zero, on
out-of-sample data that played no part in selection.** That is the first time
anything in this project has done so.

**THE BROKER SLIPPAGE BUDGET IS ~0.18 PIPS PER LEG.** Under that, the strategy
holds Sharpe > 0.5. Over it, it does not. That is now the entire question.

**What is still unknown, stated plainly:**
- 0.345 pips is the RAW MARKET cost on Dukascopy's ECN feed. A retail broker
  marks this up, and the markup varies enormously between "raw spread +
  commission" and "all-in" pricing. Broker choice may matter more than
  everything in this repo.
- Latency, requotes and rejection are unmeasured and unmeasurable from
  historical data.
- The effect is real and undecayed over 21 years, but it is a KNOWN published
  effect; forward crowding remains a risk history cannot price.

**NEXT STEP: live micro-lot execution (0.01 lots, ~$0.10/pip) to measure broker
slippage against the 0.18 pip/leg budget.** A demo account cannot answer this —
demo servers fill at the quoted price, flattering precisely the number we need.
A year of daily trades at micro size risks perhaps $20-30 total, which is the
cost of an experiment rather than a trading decision.

### 0.0 Extend history to 2005 (session 15)

**A 2005-2014 mid-price pull is RUNNING.** Seven majors, 1h, ~10 extra years.
Idempotent, so if it dies just re-run the same command and it resumes:

```bash
for s in EURUSD GBPUSD USDJPY AUDUSD USDCHF USDCAD NZDUSD; do
  python -c "
from datetime import date
from src.logging_setup import setup_logging
from src.ingest.fetch_dukascopy import pull_symbol, OFFER_SIDE_MID
setup_logging()
pull_symbol('$s','1h',date(2005,1,1),date(2014,12,31),offer_side=OFFER_SIDE_MID)
"
done

# verify (expect 22 partitions/pair once 2005-2026 is complete)
for d in data/raw/*/1h_mid; do echo "$d: $(ls $d/*.parquet | wc -l)"; done
```

Availability probed before starting: 2005 onward returns data, 2003 returns
nothing. Some pairs may start later than 2005 — the fetcher logs "returned no
rows, not writing" and moves on, so short histories are handled.

**WHY — this is the one thing that changes the arithmetic.** Proving a Sharpe of
0.5 is real needs ~17.3 years (s0.5 trap 3). We have 11.5, which is why every
confidence interval in this project has straddled zero. 2005-2026 gives ~21
years, crossing that threshold for the first time.

**WHAT IT IS FOR — and what it is NOT for.** It cannot make an edge bigger. The
h002 holdout gave 1.025 pips gross against 0.80 pips cost, and no quantity of
data changes that. The questions worth asking of the extra decade are:

1. **Is the London-open effect stable across 21 years and two market
   structures?** Surviving the GFC, the rise of algo trading, and the spread
   compression of the 2010s would be real knowledge.
2. **How much has it decayed?** EURUSD retail spreads were 2-3 pips in 2005 vs
   0.30 now. If the gross effect was much larger then and has shrunk toward the
   cost floor, that measures the crowding story directly — and says whether to
   expect further shrinkage.

Note the corollary: at 2005-era spreads the strategy would NOT have been
tradeable, so treat old data as evidence about SIGNAL EXISTENCE and STABILITY,
never about tradeability.

**Also planned with it: nested walk-forward.** The h002 selection picked the
hour by looking at aggregate stats over the whole research period, then tested
once. Nested selection — choosing inside each training window and testing on the
next — is stricter and yields ~10-15 honest out-of-sample tests instead of one.
That is the correct way to keep iterating now the holdout is spent.

### 0.1 Mid-price milestone (session 14 — SUPERSEDED as "next action" by s0.0b-e)

> **THE CURRENT NEXT ACTION is in s0.0e:** instrumented live micro-lot execution
> to measure broker slippage against the ~0.18 pip/leg budget. Everything below
> in this subsection is the session-14 mid-price verification, kept as the
> record of how the candidate was confirmed on 2015-2026 before the 2005
> extension (s0.0b) settled it over 21 years.

**The mid-price pull is COMPLETE** — all 7 pairs, 12/12 partitions,
**501,772 bars** (2015-2026 window; full history is 2005-2026, see s0.0b),
zero validation errors.

Measured median spreads (from the new `spread` column, strictly better than the
old hourly-median profile): EURUSD 0.30, USDJPY 0.40, GBPUSD 0.90, AUDUSD 1.00,
USDCHF 1.00, NZDUSD 1.10, USDCAD 1.20 pips.

**Mid-price verification: DONE (session 14). The candidate holds.**

| data | scope | t | p | gross pips |
|---|---|---|---|---|
| bid | all 7 | 3.282 | 0.00105 | 1.282 |
| MID | all 7 | 3.280 | 0.00106 | 1.284 |
| bid | EURUSD | 5.872 | <1e-6 | 2.856 |
| **MID** | **EURUSD** | **5.912** | **<1e-6** | **2.876** |

Bid and mid are indistinguishable, which is the signature of a real flow effect
(contrast the rollover artifact: t = 7.96 bid vs 1.57 mid). All four planned
attacks are now complete.

**Robustness diagnostics: DONE (session 14). All five support the candidate.**

These are DIAGNOSTICS, not searches — they ask whether the existing candidate is
coherent, not which variant scores best — so they do not add to the selection
burden the way another sweep would.

1. **No decay.** 8/8 years positive: 3.85, 3.46, 3.06, 2.75, 2.26, 0.73, 2.12,
   4.79 pips. First half 3.28 vs second half 2.47, difference p = 0.41; linear
   trend -0.09 pips/yr, p = 0.67. No detectable weakening — though the power to
   detect modest decay is low, and 2020 is nearly flat (0.73).
2. **Coherent session shape, not a lone spike.** By London hour:
   4h: -0.00, 5h: 0.76, 6h: 0.77, 7h: 1.01, **8h: 2.88**, 9h: 1.34, 10h: 1.08,
   11h: 0.74, 12h: -0.83. It rises into the open, peaks, decays, then reverses.
   A statistical fluke would have random neighbours.
3. **Coherent holding profile.** 1h 0.74, 2h 0.83, 3h 1.70, **4h 2.88**, 5h 2.05,
   6h 1.92, 12h 1.23. Accumulates through the London morning then gives back —
   consistent with a flow story, and not a knife-edge (3h and 5h both work).
4. **NOT driven by outliers — the strongest result.** Trimming the tails makes
   it STRONGER: t = 5.91 untrimmed, 6.52 at 1%, 7.63 at 5%, **8.86 at 10%**.
   Median +2.41 pips, 55.7% of days positive. A spurious effect collapses under
   trimming; this one improves, because trimming removes noise rather than
   signal.
5. **Present every weekday.** Mon 1.12, Tue 3.38, Wed 4.05, Thu 2.36, Fri 3.49
   pips — all positive, Monday weakest.

**Next action: pre-register and take the SINGLE holdout look.** This is
irreversible and is the owner's decision — see s0.3 for the prediction to
register (shrink for selection: the hour was chosen from a six-row table and
EURUSD because it is the only pair whose spread the effect clears).

To re-pull anything later, the fetcher is idempotent — `covers_range()` skips
partitions that already cover their year and re-fetches ones that do not, so
re-running is always safe:

```bash
for s in EURUSD GBPUSD USDJPY AUDUSD USDCHF USDCAD NZDUSD; do
  python -c "
from datetime import date
from src.logging_setup import setup_logging
from src.ingest.fetch_dukascopy import pull_symbol, OFFER_SIDE_MID
setup_logging()
pull_symbol('$s','1h',date(2015,1,1),date(2026,7,1),offer_side=OFFER_SIDE_MID)
"
done

# verify
for d in data/raw/*/1h_mid; do echo "$d: $(ls $d/*.parquet | wc -l)/12"; done
```

### 0.2 Where the project stands

**One validated structural edge; its tradeability rests on one unmeasured
execution number.** (This supersedes the earlier "no validated edge" framing,
which held through session 11 and was overturned by the 21-year replication in
s0.0b.)

Ten sessions of feature engineering produced essentially nothing: a screen of
82 price/carry/strength features found **1 nominal hit at p < 0.05 where chance
gives 4.1** — fewer than randomness produces. Momentum, mean-reversion, RSI,
ATR, carry, rates, currency strength and dispersion are all rigorously null,
not merely unproven.

The single survivor is **not a price pattern**. It is intraday seasonality at
the London open — a structural, flow-driven regularity, now confirmed at
p = 1e-9 over 21.5 years and independently replicated on 2005-2014 data that
played no part in finding it (s0.0b). That is the signpost: signal in
retail-accessible FX lives in how the market is ORGANISED (sessions, fixes,
expiries), not in transforms of past prices.

At the tick-MEASURED cost of 0.345 pips it earns net Sharpe 0.636 out of sample
with a CI excluding zero (s0.0e). The only open question is whether a live
broker's slippage stays within budget — a question bar data cannot answer.

### 0.3a HOLDOUT RESULT (h002, spent 2026-07-19) — REAL BUT TOO SMALL TO TRADE

**The holdout is now SPENT.** One look taken, logged, access count = 1.

| metric | research | predicted | **HOLDOUT** |
|---|---|---|---|
| gross pips/trade | 2.876 | ~2.0 | **1.025** |
| t-statistic | 5.912 | ~3.8 | **1.707** |
| p-value | <1e-6 | — | **0.088** |
| days positive | 55.7% | — | **54.9%** |
| median pips | 2.413 | — | **1.888** |
| net Sharpe | 0.873 | 0.50 ± 0.40 | **0.197** |
| net Sharpe CI | [0.196, 1.588] | — | **[-0.725, 1.204]** |
| net pips/trade | ~2.08 | — | **0.225** |
| annual return | 3.4% | — | **0.52%** |

Years: 2023 +1.392, 2024 +0.720, 2025 +1.618, 2026 (partial) **-0.290** pips.

**`evaluate_against_prereg` returned CONFIRMED. Do not take that at face value —
the substantive answer is that the strategy FAILS.**

What actually happened, honestly:

- **The effect is probably real.** It replicated in direction and shape with
  striking consistency: 54.9% of days positive against 55.7% in research, median
  +1.89 pips against +2.41. That is not what noise looks like.
- **But it is about HALF the research magnitude** (1.03 vs 2.88 pips), it is
  **not statistically significant out of sample** (t = 1.71, p = 0.088), and
  its CI straddles zero.
- **After costs it is economically negligible.** 0.225 pips per trade x 249
  trades = 0.52% per year unlevered, with a -4.1% drawdown. The 0.80 pip round
  trip eats 78% of the gross edge.
- **It fails s8 criterion 1** (net Sharpe 0.197 vs a 0.5 bar).

**A SECOND FLAW IN THE PRE-REGISTRATION FRAMEWORK, now visible.** Session 5
fixed it rubber-stamping degenerate results. It still rubber-stamps results that
land inside a WIDE TOLERANCE while failing the success criteria: 0.197 is inside
0.50 ± 0.40, so the verdict says CONFIRMED even though the strategy is not worth
trading. `evaluate_against_prereg` should also check the s8 bar, and tolerance
should be tight enough that "confirmed" implies "worth acting on".

**And the calibration lesson: my shrunk prediction was still too optimistic.**
I shrank 0.873 -> 0.50 for selection; the truth was 0.197. Selection bias plus
regression to the mean cost more than a 40% haircut — closer to 75%. Any future
pre-registration should shrink harder than feels reasonable.

**Verdict: the London-open effect is real, small, and not tradeable at retail
cost.** It is the correct answer to the question we asked, and it is a negative
one for trading purposes.

### 0.3 THE CANDIDATE (SPENT — see 0.3a for the holdout outcome)

**Short EURUSD at 08:00 London local time, hold 4 hours.**

A pure calendar rule — **no fitted parameters**, nothing to overfit but the
choice of hour, which came from a corrected screen.

| metric | value |
|---|---|
| Gross edge | +2.856 pips per trade |
| Round-trip cost | ~0.80 pips (measured) |
| Net edge | ~+2.06 pips |
| t-statistic | **5.872** (EURUSD, strictly non-overlapping) |
| p-value | < 0.00001 |
| Independent observations | 2,076 |
| Net Sharpe (earlier 4h-grid backtest) | 0.873, CI [0.196, 1.588] |
| Years positive | 8 / 8 |

**Survived four independent attacks:**
1. Mid-price check — identical on bid and mid (t 1.976 vs 1.964), so not a
   bid-side quote artifact, unlike the rollover "signal" which collapsed from
   t = 7.96 to 1.57.
2. Overlap correction — degrades modestly (3.68 -> 3.28) rather than
   collapsing, unlike COT which went from -3.75 to -0.74.
3. Bonferroni over ~180 configurations.
4. Stability — every one of 8 years positive.

**Still to do before it can be believed:** re-verify on mid data (pending the
pull above), then pre-register with a SHRUNK prediction (~0.5 Sharpe, not
0.873, because the hour was selected from a table) and take the single holdout
look.

### 0.4 Data inventory (all under data/raw/, gitignored)

| what | where | span | status |
|---|---|---|---|
| EURUSD M15 bid | `EURUSD/15min/` | 2015-2026, 286,701 bars | complete |
| 7 majors 1h bid | `<PAIR>/1h/` | 2015-2026, ~71,700 each | complete |
| 7 majors 1h MID | `<PAIR>/1h_mid/` | 2015-2026, 501,772 bars | complete, 0 errors |
| FRED rates | `macro/` | DGS2, DGS10, DFF, ECBDFR | complete |
| CFTC COT | `cot/` | 2014-2026, 654 wks x 7 ccy | complete |
| Spread profile | `spread/EURUSD_hourly.json` | 2019-2022 measured | complete |

Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD, NZDUSD.

### 0.5 Methodology traps we have actually hit

Every one of these produced a plausible-looking false result before being
caught. They are the most valuable thing in this repo.

1. **Overlapping forward returns fabricate significance.** An H-bar forward
   return sampled every bar overlaps (H-1)/H; naive standard errors inflate by
   ~sqrt(H). COT scored t = -3.75 overlapping and t = -0.74 non-overlapping.
   Blocking by date fixes CROSS-SECTIONAL correlation and does nothing for
   serial overlap — different problems, both need handling.
   Guarded by `scoring.overlap_warning`.
2. **Bid-only prices create fake signal at illiquid hours.** The bid gaps and
   recovers at rollover while mid barely moves. t = 7.96 on bid, 1.57 on mid.
   **Always use mid.**
3. **Sharpe cannot resolve anything at this sample size.** Proving S = 0.5 is
   real needs ~17 years; we have 11.5. Use per-prediction tests
   (`signed_return_test`) to establish signal, and Sharpe only afterwards to
   ask whether it survives costs.
4. **Log loss is HALF as powerful as a directional test** for detecting a
   calibrated edge (z 1.79 vs 3.58). It grades calibration, not direction.
   Never optimise for it — it weights all observations equally while P&L
   weights by move size.
5. **A near-constant series passes `se > 0`.** Floating-point std ~1e-17 gave
   t = 6.5e16, p = 0.0 from data containing no information. Degeneracy guards
   need a threshold RELATIVE to the magnitude tested.
6. **Filters that cost more than they save.** `min_edge` compared per-bar edge
   to a full round-trip cost (~7x too strict); the session filter flattened
   positions to avoid a spread it never paid, tripling turnover. Costs are
   incurred on TRADES, not on exposure.
7. **Existence-only idempotency corrupts data.** A partition that exists but
   under-covers its year was silently skipped, nearly leaving 11 months missing
   mid-dataset. Check coverage, not existence.
8. **Prefix matching grabs the wrong instruments.** `startswith("EURO FX")`
   also matched EUR/GBP and EUR/JPY cross-rate contracts.
9. **A pre-registration framework that rubber-stamps degenerate results is
   worse than none.** h001 scored "within tolerance" while trading in 0.89% of
   bars. Confirmation now requires a clean degeneracy report.

### 0.6 Search budget

~180 configurations tested. Bonferroni alpha ~0.0003. Any new candidate must
clear the corrected threshold on the `signed_return_test` gate BEFORE its
Sharpe is taken seriously (see s8, s10).

### 0.7 What has been ruled out

- All price-derived features, unconditionally (s11 screen).
- Regime conditioning — 0 of 7 interactions significant; signals are not
  cancelling across vol regimes, they are absent (s12).
- Rates/carry as a directional predictor (s4).
- COT positioning, once overlap is corrected (s13).
- Calibration and ensembling as fixes — both made results worse; the binding
  constraint is sample size and signal strength, not model sophistication (s8).
- Structural calendar features other than sessions — month-end, quarter-end,
  option expiry all flat (s12).
- Cross-asset series (equities, gold, oil, copper, gas) as a directional LEAD
  (s16, h003). Strong CONTEMPORANEOUS co-movement (gold/EUR corr 0.40 same
  window) but ~0 as a lead; the FDR "survivors" collapse under trimming and have
  sub-50% hit rates. One unconfirmed reversal (risk-on -> haven strengthens next
  session) is left as a candidate for a fresh pre-registration, not a result.

---

## 1. Data Sources

### Primary: Dukascopy tick data
Dukascopy is the source of truth — free, ECN-sourced tick data, widely regarded
as one of the better retail-accessible feeds.

- Web tool (manual/small pulls only, not for bulk): 
  https://www.dukascopy.com/swiss/english/marketwatch/historical/
- Programmatic access (preferred — use one of these, don't scrape the web UI):
  - `dukascopy-python` (PyPI) — https://pypi.org/project/dukascopy-python/
    - `fetch()` for static historical pulls, `live_fetch()` for streaming.
    - Supports tick-level and aggregated (M1/M5/M15/M30/H1/H4/D/W/M) intervals.
  - `dukascopy-node` (Node/CLI, if we end up wanting a non-Python fetch step) —
    https://www.dukascopy-node.app/
    - Historical data back to 1990s-2000s depending on instrument, 1000+
      instruments (FX, commodities, indices, crypto).

Do NOT use forexsb.com as the primary source — it's a secondary compilation of
Dukascopy ticks (capped at 200,000 bars/series) built for a specific EA-builder
ecosystem. Fine as a sanity-check cross-reference, not as the pipeline's source
of truth.

### Secondary / cross-checks
- MetaTrader History Center (MT4/MT5) — useful for cross-validating bar
  construction against a widely-used retail platform, not primary.
- OANDA v20 REST API — free for account holders, useful if we want a second
  independent data vendor to check for Dukascopy-specific artifacts.

---

## 2. Project Structure

**AS BUILT** (the sketch this section originally contained is superseded).
Run everything from the repo root; `pyproject.toml` sets `pythonpath = ["."]`.

```
Predicto-Forex/
├── CLAUDE.md                     # this file — s0 is the entry point
├── README.md
├── pyproject.toml, requirements.txt, .gitignore
├── config/instruments.yaml       # 7 majors: pip size, spread, date range
├── preregistrations/*.json       # committed hypotheses (h001 spent, INCONCLUSIVE)
├── data/raw/                     # gitignored — see s0.4 inventory
├── logs/                         # gitignored, INCLUDING holdout_access.log (see below)
├── src/
│   ├── config.py                 # paths + instruments.yaml loader
│   ├── logging_setup.py
│   ├── run_pipeline.py           # end-to-end entrypoint
│   ├── ingest/
│   │   ├── fetch_dukascopy.py    # bid/ask/MID, retries, coverage-aware idempotency
│   │   ├── validate_raw.py       # tz/gap/duplicate/OHLC checks; exits non-zero
│   │   ├── fetch_fred.py         # DGS2/DGS10/DFF/ECBDFR, 1-day publication lag
│   │   ├── fetch_cot.py          # CFTC positioning, 6-day release lag
│   │   └── measure_spread.py     # hourly spread profile from bid vs ask
│   ├── features/
│   │   ├── build_features.py     # price features, resample_bars, assemble_dataset
│   │   ├── currency_strength.py  # cross-pair decomposition (exact to 2e-17)
│   │   └── structural.py         # DST-aware sessions, month-end, expiry
│   ├── models/
│   │   ├── baseline.py           # buy-hold, random, momentum, mean-rev, ridge
│   │   ├── calibration.py        # isotonic (imported from AFL; did not help)
│   │   ├── ensemble.py           # feature-family ensemble + agreement filter
│   │   └── cross_sectional.py    # currency-weight book, minimum-norm netting
│   ├── backtest/
│   │   ├── engine.py             # THE execution convention lives here
│   │   ├── walk_forward.py       # rolling/expanding splits with embargo
│   │   ├── costs.py              # hour-aware spread, amortized breakeven, carry
│   │   ├── metrics.py            # Sharpe/Sortino/DD + block-bootstrap CIs
│   │   ├── scoring.py            # per-prediction tests — THE POWERFUL GATE
│   │   └── holdout.py            # seal, pre-registration, access log
│   ├── sizing/kelly.py           # fractional Kelly, capped, drawdown throttle
│   └── analysis/
│       ├── feature_screen.py     # corrected screen across families/timeframes
│       ├── regime.py             # conditional + interaction tests
│       └── execution.py          # tick-level fill reconstruction (real cost)
├── docs/
│   ├── project_explainer.html    # plain-English writeup (source of truth)
│   └── Predicto-Forex-Explained.pdf  # 8pp, rendered via headless Chrome
└── tests/                        # 200 tests
```

**Known gap:** `logs/` is gitignored, so `logs/holdout_access.log` — the audit
trail of how many times the holdout has been read — is NOT version controlled
and can be silently deleted. That record is what makes pre-registration
meaningful. Consider un-ignoring it.

### 2a. Common commands

```bash
pytest -q                                    # 200 tests
python -m src.ingest.fetch_dukascopy --symbol EURUSD --timeframe 1h \
       --start 2015-01-01 --end 2026-07-01   # idempotent; add --overwrite to force
python -m src.ingest.validate_raw --symbol EURUSD --timeframe 1h
python -m src.ingest.fetch_fred                       # rates
python -m src.ingest.fetch_cot --refresh              # positioning
python -m src.ingest.measure_spread --symbol EURUSD   # hourly spread profile
python -m src.run_pipeline --symbol EURUSD --timeframe 15min \
       --train-size 20000 --test-size 5000
```

---

## 3. Environment Setup

- Python 3.11+, virtualenv or conda.
- Core deps: `dukascopy-python`, `pandas`, `numpy`, `pyarrow` (for parquet storage
  of tick/bar data — CSV will not scale), `scikit-learn`, `matplotlib`.
- Optional depending on model choice: `xgboost`/`lightgbm` for tree models,
  `torch` if we go neural, `statsmodels` for classical time series (ARIMA/GARCH
  baselines).
- Store all raw pulls as parquet, partitioned by instrument/year, not CSV —
  tick data at scale will blow up CSV read times.

---

## 4. Methodology — Non-Negotiables

These come directly from lessons learned on the DESI/Fisher pipeline work —
same discipline applies here:

1. **No lookahead bias.** Any feature must only use information available
   strictly before the timestamp it's predicting. Build a unit test that
   asserts this for every feature function.
2. **Walk-forward validation, not a single train/test split.** Use rolling or
   expanding windows; report performance variance across windows, not just a
   mean.
3. **Transaction costs from day one.** Every backtest includes spread,
   estimated slippage, and swap/rollover — never report a "gross" Sharpe
   without a "net" Sharpe next to it.
4. **Kill hypotheses cheaply before scaling up.** Validate on a short
   date range / single pair before running the full multi-instrument,
   multi-year backtest. Mirrors the "kill hypotheses cheaply before
   committing HPC resources" approach used on Sciama.
5. **Position sizing via fractional Kelly**, reusing/adapting the Kelly
   criterion module from the sports betting pipelines. Cap max position size
   and max drawdown explicitly — don't let the model size itself unbounded.
6. **Explicit uncertainty flagging.** Every backtest result should report a
   confidence interval or resampled distribution (e.g. block bootstrap on
   returns), not a single point estimate.
7. **Paper trade before live.** No real capital until a strategy has passed
   walk-forward validation AND a live paper-trading period matching backtest
   expectations within a pre-registered tolerance.

---

## 4a. Suggested Skeleton — `src/ingest/fetch_dukascopy.py`

This is a starting skeleton, not a finished spec. If you (Claude Code / whatever
model is running this) have a better idea for structuring the ingestion layer —
different retry strategy, async fetching, a different on-disk layout, better
error handling, whatever — improve it. This is a sketch to save you a blank
page, not a constraint to follow literally.

```python
"""
fetch_dukascopy.py

Wraps dukascopy-python to pull historical bar/tick data and write it to
partitioned parquet under data/raw/. Should be safe to re-run (idempotent —
skip or overwrite partitions that already exist, don't duplicate).
"""

from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD  # etc.

RAW_DATA_DIR = Path("data/raw")

# Map friendly symbol names -> dukascopy_python instrument constants.
# Extend as needed; keep in sync with config/instruments.yaml.
INSTRUMENT_MAP = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    # "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
    # "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
}


def fetch_range(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
) -> pd.DataFrame:
    """
    Fetch one symbol/timeframe/date range from Dukascopy with retries.
    Raises after max_retries exhausted — caller decides whether to skip
    or halt the whole pull.
    """
    instrument = INSTRUMENT_MAP[symbol]

    for attempt in range(1, max_retries + 1):
        try:
            df = dukascopy_python.fetch(
                instrument=instrument,
                interval=timeframe,   # e.g. dukascopy_python.INTERVAL_MIN_15
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=start,
                end=end,
            )
            return df
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = retry_backoff_s * attempt
            print(f"[{symbol}] attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)


def write_partition(df: pd.DataFrame, symbol: str, timeframe: str, year: int) -> Path:
    """Write one symbol/timeframe/year partition to parquet. Idempotent."""
    out_dir = RAW_DATA_DIR / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.parquet"
    df.to_parquet(out_path, index=True)
    return out_path


def pull_symbol(
    symbol: str,
    timeframe: str,
    start_year: int,
    end_year: int,
    overwrite: bool = False,
) -> None:
    """
    Loop year-by-year (keeps memory bounded and gives natural checkpoints
    for a multi-year pull) and write each year's partition.
    """
    for year in range(start_year, end_year + 1):
        out_path = RAW_DATA_DIR / symbol / timeframe / f"{year}.parquet"
        if out_path.exists() and not overwrite:
            print(f"[{symbol}][{timeframe}][{year}] already exists, skipping")
            continue

        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        df = fetch_range(symbol, start, end, timeframe)
        path = write_partition(df, symbol, timeframe, year)
        print(f"[{symbol}][{timeframe}][{year}] wrote {len(df)} rows -> {path}")


if __name__ == "__main__":
    # Smoke test per the session plan: small range, one pair, before
    # committing to a full historical pull.
    pull_symbol("EURUSD", timeframe="15min", start_year=2026, end_year=2026)
```

Known gaps in this skeleton, deliberately left for whoever builds it out:
- No handling yet for tick-level pulls specifically (higher volume, probably
  needs day-level partitions instead of year-level).
- No timezone normalization step — needs to happen either here or in
  `validate_raw.py`, decide which and document it.
- No config-file wiring (`config/instruments.yaml` isn't read yet — this
  hardcodes one symbol as a starting point).
- No logging framework — just prints. Fine for a first pass, swap in proper
  logging once the pipeline stabilizes.

---

## 5. Open Questions / To Decide Early

- Which instrument(s) to start with — majors only (EURUSD, GBPUSD, USDJPY) or
  broader universe? Recommend starting narrow (1-2 pairs) to validate pipeline
  before scaling.
- Target holding period / frequency: intraday (M1-M15), swing (H1-H4), or
  daily+? This drives both feature engineering and how seriously transaction
  costs bite.
- Model class: classical statistical (mean-reversion/momentum signals,
  GARCH-based vol models) vs. ML (gradient boosting on engineered features) vs.
  something closer to the SBI-style approach from the DESI work. Worth
  starting with strong simple baselines before anything complex.
- Broker for eventual paper/live testing (affects realistic cost assumptions
  in the backtest) — not needed yet, but backtest cost assumptions should
  target a specific broker's actual spread/leverage terms once chosen.

---

## 6. Session Log

_(Claude Code: append a dated entry here each session — what was done, what
was found, what's next. Same pattern as the Sciama Fisher pipeline handoff
file.)_

### 2026-07-19 — Project initialized
- Decided on Dukascopy (via `dukascopy-python`) as primary data source.
- Decided against forexsb.com as primary source (secondary/compiled data,
  200k-bar cap) — useful only as a cross-check.
- Repo structure and methodology non-negotiables drafted (see above).
- Next: scaffold repo, write `fetch_dukascopy.py`, pull a small test range
  (e.g. 1 month EURUSD M15) and run `validate_raw.py` gap/duplicate checks
  before committing to a full historical pull.

### 2026-07-19 — Skeleton built, end-to-end smoke run green
Scaffolded the full repo per s2 and wired it end to end. 52 tests passing.

**Built:** ingest (`fetch_dukascopy`, `validate_raw`), features, baselines,
walk-forward splitter, costs, metrics, Kelly sizing, backtest engine, and
`src/run_pipeline.py` as the end-to-end entrypoint. Config is wired through
`config/instruments.yaml` via `src/config.py`; logging via `src/logging_setup.py`.

**Decisions made (previously open):**
- **Timezone normalization lives in ingest, not validation.** `fetch_dukascopy`
  emits a tz-aware UTC index named `timestamp`; `validate_raw` asserts that
  rather than fixing it, so a silent repair can't mask a bad pull.
- **Tick partitions are day-level** (`data/raw/SYM/tick/YYYY/YYYY-MM-DD.parquet`),
  bars stay year-level. An instrument-year of ticks won't fit in memory.
- **`validate_raw` reports, never repairs**, and exits non-zero on ERROR so it
  can gate CI. Data gaps are warnings; structural defects are errors.
- **Execution convention: `strategy_return[t] = position[t-1] * market_return[t]`.**
  Documented at the top of `engine.py` — this is the project's easiest way to
  fake a good backtest, so it's guarded by a test.
- Note `dukascopy_python.fetch()` already retries internally and paginates via
  `limit`; our wrapper's retry loop is only for whole-call failures.

**Both critical invariants are mutation-tested**, not just asserted:
- Making a feature window centered → 5 lookahead tests fail. ✓
- Dropping the position shift in `engine.py` → a stale (already-realized)
  signal reports 91% hit rate and Sharpe ~180, and the guard test fails. ✓

**Smoke run** (1 month EURUSD M15, 2028 bars, validation clean, 5 folds):
all baselines ran; `random` went from −6.0 gross Sharpe to −20.8 net, which is
the check that costs are actually biting. Every CI straddles zero except the
ones that are clearly negative — as expected on one month of data. **No signal
is claimed here; this run only proves the plumbing is correct.**

**Next:**
- Pull real history (2015→now, EURUSD M15) and re-run — current numbers are
  from one month and are statistically meaningless.
- Decide holding period (s5, still open) — it drives feature windows and how
  hard costs bite. `breakeven_edge()` in `costs.py` is the cheap first filter.
- `BARS_PER_YEAR` and `TIMEFRAME_MINUTES` are hardcoded estimates; derive them
  from the actual FX calendar once there's real multi-year data to measure.
- Swap/rollover is a flat annual rate in `CostModel` — replace with real
  per-pair carry once a broker is chosen (s5).
- No model beyond ridge yet; baselines must be beaten before adding complexity.

### 2026-07-19 (later) — Multi-year data pulled; all baselines fail net of costs

**Data:** EURUSD M15, 2015-01-01 → 2026-07-01. 286,701 bars, 12 year-partitions,
validation clean (0 errors). Per-year counts are consistent (~24,900-25,000);
price range 0.954-1.255 spans the 2022 parity break, so the data is genuine.
The 18 flagged gaps are all New Year / Christmas closures — expected, not defects.

**Bug found and fixed — partition idempotency was existence-only.** The 1-month
smoke-test `2025.parquet` (0.09 MB vs ~0.8 MB for full years) was silently
skipped by the full pull, leaving 11 months missing from the middle of the
dataset. `covers_range()` now checks a partition's actual span against the
requested range and re-fetches if it under-covers, with a 4-day tolerance for
holiday edges. Verified live: re-running the pull re-fetched only 2025 and left
the other 11 partitions untouched. Regression test in
`tests/test_partition_coverage.py`. 60 tests passing.

**Result: no baseline shows tradeable edge. Costs dominate completely.**

Initial run (train 20k / test 5k, ~53 folds), net Sharpe: buy_and_hold -0.11,
mean_reversion -1.51, momentum -3.66, ridge -9.05, random -15.93.

Ridge is the instructive case: **+3.17 gross Sharpe → -9.05 net**. Diagnosis
confirmed this is real, not a cost-model bug:
- predicted edge 0.17 pips vs **1.0 pip breakeven** — signal is 6x too small
  to pay for its own trading
- mean holding period **1.36 bars** (~20 min); it flips almost every bar
- turnover 17,947 units/yr x 0.5bp = 89.7% drag, matching the measured 90.7%
  (i.e. the arithmetic is internally consistent)
- gross +24.4%/yr is real but buried under 90.7% of costs

**Horizon sweep** (15m/1h/4h/1d/1w x ridge/momentum, with `min_edge` set to
breakeven). Setting `min_edge=breakeven_edge()` is a large lever — it cut 15m
ridge turnover 17,947 → 66 and net Sharpe -9.05 → -0.04. But **no combination
produced a net Sharpe whose CI excludes zero.** Longer horizons raise the
edge/breakeven ratio (5.4x at 1w) without improving gross Sharpe, i.e. ridge
isn't finding anything real — it's just predicting less often.

**Do not be fooled by 15m momentum** (net Sharpe 0.303, CI [0.303, 0.607]).
`fold_std` is NaN and turnover is 0.187 units/YEAR — it trades about once every
five years, so that Sharpe is computed on a near-empty return series from a
single fold. Degenerate artifact, not a signal. Any future leaderboard should
disqualify runs with near-zero turnover or NaN fold variance rather than
ranking them.

This is the pipeline working correctly: it is refusing to manufacture signal
from naive features on a liquid major. That is the expected outcome and the
reason the cost model and CIs were built in from day one.

**Next:**
- The bottleneck is signal quality, not model complexity. An xgboost on these
  same features will very likely also fail — do not reach for it first.
- Most promising directions, in order: (a) cross-pair / carry / rates-differential
  features rather than pure price-derived ones; (b) a regime filter that trades
  only in specific sessions or vol states, cutting turnover hard; (c) genuinely
  longer horizons (multi-day swing) where the 1-pip cost floor stops dominating.
- Add a turnover/degeneracy guard to `run_pipeline` so artifacts like the
  momentum case above can't reach a leaderboard.
- `min_edge` should probably default to `breakeven_edge(cost_model)` rather than
  0.0 — sizing on sub-breakeven predictions is never rational.
- Still open from s5: holding period is now partly answered (1-bar M15 is
  definitively dead), but no horizon tested is alive either.

### 2026-07-19 (later still) — Cost surface measured; horizon is the binding constraint

**Required information coefficient (IC) by horizon**, EURUSD, using measured
1-pip breakeven against realized vol at each horizon:

| horizon | vol (pips) | required IC |
|---|---|---|
| 15m | 5.2 | **0.192** |
| 1h | 10.2 | 0.098 |
| 4h | 20.3 | 0.049 |
| 1d | 48.6 | 0.021 |
| 1w | 107.9 | 0.009 |

A realistic IC for a good FX signal is 0.02-0.05. **At M15 we needed 0.19 —
roughly 5-10x beyond what is achievable.** We were not failing to find signal;
we were fishing where no achievable signal could have paid.

Notably our M15 ridge achieved IC ~0.033 (0.17 pips edge / 5.2 pips vol), which
is a *respectable* number — it lost only because the hurdle was 0.19. But at 1d
ridge's gross Sharpe went negative, so these price-derived features have no
skill at longer horizons. **Both changes are needed: longer horizon AND better
features. Neither alone is sufficient.**

**Measured spreads (2024, bid vs ask, 1h bars) — our 0.6-pip assumption was wrong:**

| pair | median spread | 1d vol (pips) | breakeven | required IC @1d |
|---|---|---|---|---|
| USDJPY | 0.60 | 100.9 | 1.00 | **0.010** |
| EURUSD | 0.20 | 39.0 | 0.60 | 0.015 |
| GBPUSD | 0.80 | 48.3 | 1.20 | 0.025 |

- EURUSD median spread is **0.20 pips, not the 0.6 we assumed** (0.6 is ~p90).
  The cost model is ~3x too pessimistic at median, ~right at the tail.
- **Spread varies ~7x by hour**: EURUSD 0.20 @14h UTC (London-NY overlap) vs
  1.45 @21h (rollover). GBPUSD 0.70→2.90, USDJPY 0.50→4.20. A flat spread
  constant is simply wrong; `CostModel` should take an hour-of-day profile,
  and 21h UTC should probably be excluded from trading entirely.
- Caveat on USDJPY: 2024 was an exceptional vol regime (BoJ intervention, carry
  unwind). Re-measure across 2015-2026 before treating 0.010 as durable.

**Pair choice matters far less than horizon.** Best-to-worst pair spans
0.010-0.025 required IC (2.5x). Horizon spans 0.009-0.192 (21x). Chasing
"less efficient" exotic pairs is a trap — spread grows faster than opportunity.

**Feature direction — the real bottleneck.** Every current feature is a
transform of a single price path (momentum, z-score, RSI, ATR, range position),
so they carry no information the price series doesn't already contain. Ranked
by expected value:
1. **Rates/carry differential** (2y yield spread, policy rates; free from FRED).
   The most robustly documented FX return driver. Slow-moving, suits daily+.
2. **Cross-pair USD factor** — decompose EURUSD into EUR-strength and
   USD-strength via a basket (GBPUSD, AUDUSD, USDJPY...). Genuinely new
   information relative to EURUSD alone; FX is a relative-value market.
3. **COT positioning** (CFTC, weekly, free) — speculative positioning extremes
   are a documented reversal signal.
4. **Measured spread / microstructure** — now that we can pull ask side and
   ticks: spread as a regime feature, tick-volume imbalance, trade intensity.
5. **Macro event calendar** (NFP, FOMC, ECB) — returns around scheduled events
   are structurally different; worth a flag or an exclusion.

### 2026-07-19 (session 4) — FRED rates wired in; carry financing fixed

**Built:** `src/ingest/fetch_fred.py` (no API key needed — public fredgraph CSV
endpoint), `resample_bars()` in build_features, `with_macro=True` on
`assemble_dataset`, and real signed carry in `apply_costs`. 91 tests passing.

**Usable FRED series (daily, non-revised):** `DGS2`, `DGS10`, `DFF`, `ECBDFR`.
Euro-area and JP yield series on FRED are monthly AND lag ~6 months in
publication (`IRLTLT01EZM156N` last point 2026-01-01), so they are unusable
for a daily strategy. `INTDSRJPM193N` died in 2017. EURUSD differential is
therefore `ECBDFR - DFF`, both daily policy rates, symmetric.
Sanity-checked against history: ECB deposit rate spans -0.50% (NIRP era) to
4.00% (2023); Fed funds 0.04% to 5.33%. Correct.

**Alignment causality** is mutation-tested (`tests/test_fred_alignment.py`):
switching ffill→bfill, or dropping the 1-day lag, both fail the guards. FRED
serves latest-revised data, so only market/policy rates are used —
do NOT add GDP/CPI/payrolls without switching to ALFRED vintages.

**Finding 1: rate features are NOT a directional predictor of spot.**
`corr(f_rate_diff, fwd_ret) = -0.005` against a required IC of 0.021. Adding
the 6 macro features made ridge strictly worse (4h 0.083 → -0.189 net Sharpe;
1D -0.002 → -0.468) — extra noise, no signal. Momentum/buy-hold were
byte-identical with and without, the correct consistency check.

**Finding 2 (my error, now fixed): the first carry test was mis-specified.**
Carry earns the differential as INCOME; our target was spot return only and
`CostModel` charged a flat symmetric 1%/yr swap. That makes carry untestable
by construction — the thing being harvested never entered the P&L. `apply_costs`
now takes `carry_annual` (signed, real differential) and returns a `carry`
column. Sign convention: EUR rates average 1.45% BELOW USD, so long EURUSD
bleeds and short earns.

**Finding 3: a candidate, NOT a validated edge.** Daily EURUSD, momentum,
real carry, measured 0.2-pip spread: net Sharpe **0.584, CI [0.102, 1.121]** —
the first CI in this project that excludes zero. Turnover 68/yr, 10 folds, so
not degenerate like the earlier artifact.

**Do not act on this yet. Two serious caveats:**
1. **Multiple comparisons.** We have now tested ~33 configurations (5 baselines,
   10 horizon-sweep, 12 macro/timeframe, 6 carry). At 95% confidence, ~1.7 are
   expected to clear zero BY CHANCE. One CI barely above zero (lower bound
   0.102) is exactly what noise predicts. This needs a Bonferroni-style
   correction or, better, a held-out period never touched during search.
2. **Most of the improvement is a cost-model change, not signal.** Momentum's
   carry P&L is -0.005 (≈zero). It went 0.465 → 0.584 mainly because real
   signed financing replaced a flat 1%/yr charge that had been applied
   regardless of direction. More accurate, but it flatters results by removing
   a systematic drag rather than by finding anything.

**Next:**
- **Carve out a true holdout** (e.g. 2023-2026) and do not run a single
  backtest on it until a hypothesis is pre-registered. Everything to date has
  seen all the data; the ~33 configurations have contaminated the full sample.
- Apply a multiple-comparisons correction to any leaderboard.
- Add the degeneracy guard (near-zero turnover / NaN fold variance) that the
  earlier momentum artifact motivated — still not done.
- `min_edge` should default to `breakeven_edge(cost_model)` — still not done.
- Cross-pair USD factor is the untested feature with the best prior; rates are
  now known not to work directionally on their own.

### 2026-07-19 (session 5) — Holdout sealed; h001 came back INCONCLUSIVE

**Built:** `src/backtest/holdout.py` (seal date, pre-registration, append-only
access log, Bonferroni helper), `BacktestResult.degeneracy`, and `min_edge`
now defaults to `breakeven_edge(cost_model)`. 90 tests passing.

**Holdout:** sealed at **2023-01-01**. Research 2015-01-02..2022-12-31 (2501
daily bars), holdout 2023-01-02..2026-07-02 (1095 bars, 30% of sample).
Access is logged to `logs/holdout_access.log`; every read is counted and
reported alongside any result.

**Research-only leaderboard (pre-2023)** picked `naive_momentum`: net Sharpe
0.794, CI [0.107, 1.626], 6 folds, turnover 67/yr, non-degenerate. The new
degeneracy guard correctly flagged `buy_and_hold` (turnover 0.173/yr).

**h001 pre-registered** at predicted net Sharpe 0.40 ± 0.50, then evaluated
once. Result: **INCONCLUSIVE, not confirmed.**

Observed Sharpe 0.552 — nominally inside tolerance — but the degeneracy guard
fired on all of: in-market only **0.89% of bars**, 1 of 3 folds producing a
finite Sharpe, NaN across-fold variance, and a total net return of **0.7% over
3.5 years**. The strategy essentially stopped trading in the holdout: momentum
predictions fell below breakeven almost everywhere once `min_edge` defaulted
to the real threshold. A Sharpe computed on ~1% exposure is not evidence.

**This exposed a flaw in the framework itself.** `evaluate_against_prereg`
originally returned `confirmed: true` for this — a pre-registration system that
rubber-stamps a degenerate result is worse than none, because it launders noise
as validation. It now requires a clean degeneracy report and returns an explicit
verdict of CONFIRMED / REFUTED / INCONCLUSIVE (degenerate). Regression test in
`tests/test_holdout.py::test_degenerate_result_cannot_be_confirmed`.

**Important caveat on this holdout:** ~33 configurations were searched over the
FULL 2015-2026 sample in sessions 3-4, before the seal existed. The formal
selection for h001 used research data only, but the analyst's priors had
already seen the holdout. This holdout is therefore *partially burnt* and is
best-effort, not pristine. Genuinely clean validation now requires forward
paper trading (s4.7) on data that does not yet exist.

**Status: no validated edge. h001 neither confirmed nor refuted — it was not
really tested.** The honest read of five sessions is that naive price-derived
features on EURUSD do not clear costs at any horizon tried, and the one
candidate that survived research selection evaporated when the breakeven filter
was applied properly.

**Next:**
- h001 is spent as an id. A retest needs a NEW hypothesis id and should size
  the position differently (e.g. lower `min_edge`, or a signal with enough
  amplitude to clear breakeven) so the holdout actually gets exercised.
- Cross-pair / currency-factor features (see s7 below) are the best remaining
  untested idea and should be developed on research data only.
- Consider a second seal (e.g. 2025+) reserved for whatever comes after the
  cross-pair work, since 2023-2026 is now partially spent.

---

## 7. On Multi-Currency (Triangular) Relationships

Asked in session 5: "is it worth trying to work the relationship between three
currencies?" Short answer: **yes, but not as arbitrage.** Three distinct ideas
get conflated here and they have very different prospects.

**1. Triangular arbitrage — dead. Do not pursue.**
EURUSD x USDJPY = EURJPY must hold by no-arbitrage. Any deviation is closed by
co-located HFT in microseconds, and the deviation is smaller than our 0.2-0.6
pip spread before we could act. We measured retail costs at ~0.6-1.2 pips
round trip; the mispricing is a fraction of that. This is structurally
unavailable to a retail research pipeline and should not be attempted.

**2. Currency factor decomposition — the best untested idea we have.**
EURUSD is a *ratio*: a move can come from EUR strength or USD weakness, and the
single price series cannot distinguish them. With a basket (EURUSD, GBPUSD,
AUDUSD, USDJPY, USDCHF...) you can decompose returns into per-currency strength
factors — if every USD pair moves together it is a dollar move, not a euro move.

This is genuinely *exogenous information relative to the EURUSD price path*,
which is exactly what every feature we have built so far lacks (see session 4:
all price-derived features are transforms of one series and carry no new
information). It is the same reason rates were worth trying, but with a much
stronger prior because it is mechanically related to the thing we are
predicting rather than macro-linked to it.

**3. Cross-sectional ranking — the well-documented academic version.**
The FX factor literature (carry, momentum, value) finds these work markedly
better *cross-sectionally* — rank currencies, go long the top and short the
bottom — than as time-series signals on a single pair. Our entire project so
far has been time-series on one pair, i.e. the weaker formulation.

**Cost warning, which is the catch.** Trading N pairs multiplies spread cost by
roughly N, and triangular positions are *redundant*: an EURJPY position is
already implied by EURUSD + USDJPY exposure. Expressing a view through three
legs pays three spreads for two legs' worth of exposure. Any implementation
must net exposures down to the minimum spanning set before trading. Given that
costs have killed every strategy tested so far, this is the binding constraint,
not the signal.

**Recommended approach if pursued:** pull 5-6 USD majors, build per-currency
strength factors, test cross-sectional ranking at daily horizon, net positions
to minimum legs, and develop entirely on pre-2023 research data.

---

### 2026-07-19 (session 6) — Two cost-side bugs fixed; earlier results were too harsh

**Built:** `src/ingest/measure_spread.py`, hour-aware `CostModel`
(`spread_profile`, `cost_per_unit_series`, `from_measured_profile`),
`mean_holding_bars`, `amortized_breakeven_edge`, and an `allowed_hours`
session filter in the engine. 109 tests passing.

**Bug 1 — `min_edge` was ~7x too strict.** It compared each bar's predicted
edge against the FULL round-trip breakeven, but positions persist ~7.1 bars, so
the spread is paid once per 7 bars. The correct per-bar hurdle is cost/H.
Against the amortized figure 95.7% of bars clear versus 72.4% before. This is
what silently zeroed the h001 holdout test (in-market 0.89% of bars).

**Consequence: several session-4 conclusions were biased pessimistic.** Re-run
on research data with the fix plus measured spreads, 4h ridge goes **-3.075 ->
-0.301** and 1d momentum **-0.044 -> +0.278**. The horizon sweep was penalising
strategies for costs they would not have paid.

**Measured spread profile (EURUSD, 2019-2022, ~25k hourly obs):** flat at
**0.30 pips from 06:00-19:00 UTC**, 0.40 overnight, then **1.20 / 1.50 / 0.60
at 20:00 / 21:00 / 22:00** (rollover). Cheapest-to-dearest ratio 5.0x, but only
THREE hours are expensive. Cached at `data/raw/spread/EURUSD_hourly.json`.

This corrected an earlier recommendation: session 4 suggested restricting to
12:00-16:00 UTC based on a 2024-only sample. That would have been **wrong** —
it would cut ~80% of trading opportunities for no cost benefit, because the
day is essentially flat outside rollover. The right rule is to exclude 20-22
UTC only. Measure before restricting.

**Bug 2 — the session filter's first implementation made things worse.**
Zeroing positions during blocked hours forces a flatten-and-re-enter every day.
Spread is paid on TRADES, not on exposure, so holding through the rollover is
free while closing out costs a full round trip. Measured: 4h mean-reversion
turnover 162 -> 552/yr and net Sharpe +0.380 -> +0.086. Fixed by FREEZING the
position (NaN + ffill) instead. After the fix the filter improves net Sharpe in
5 of 6 configurations and reduces turnover, as intended.

Both bugs are the same species: **a filter that costs more than it saves.**
Worth watching for in anything added next.

**Research-only leaderboard after both fixes (2015-2022):**
best is 4h mean-reversion at net Sharpe **0.451**, then 1d momentum 0.278.
Everything else is negative. **Nothing has a CI excluding zero.**

**Robustness red flag:** 1d momentum scored 0.794 with a 750/250 splitter in
session 5 but 0.278 with 600/200 here. A result that swings that much on an
arbitrary split parameter is behaving like noise, not signal. Any future
candidate should be checked across several splitter settings BEFORE it is
pre-registered.

**Next:**
- Config count is now ~50. Bonferroni alpha is 0.001; a candidate needs a
  ~99.9% interval excluding zero. Nothing is close.
- Cross-pair currency factors (s7) remain the best untested idea, and now
  inherit a correctly-specified cost model.
- Still worth agreeing an explicit stopping rule (proposed: net Sharpe > 0.5 on
  a clean holdout with turnover > 20/yr and stability across splitters, or stop
  and treat the pipeline as the deliverable).

---

## 8. Success Criteria (AGREED 2026-07-19)

A strategy is worth paper trading if and ONLY if it clears all four:

1. **Net Sharpe > 0.5** on a clean holdout, after all costs.
2. **Turnover > 20/yr** — no degenerate near-zero-trading artifacts.
3. **Stable across splitters** — must survive several train/test window sizes.
   Added because 1d momentum scored 0.794 at 750/250 but 0.278 at 600/200,
   which is noise behaviour.
4. **No degeneracy flags** from `BacktestResult.degeneracy`.

Owner has stated the goal is profitability even if it takes a long time, so
the search continues rather than stopping at the first null result. The bar
above is what converts "looks good" into "worth risking money on". It may be
revised, but ONLY before a result is seen, never after.

---

## 9. Transferable Techniques From Sibling Modules

Reviewed `PL-AFL-Module` and `PL-Tennis-Module` (2026-07-19) for approaches
worth importing. Four are genuinely relevant; one is a trap.

**9.1 Elo-style currency strength ratings (from PL-Tennis-Module).**
`ATPBetting/Python/elo_features.py` maintains a per-player rating updated after
each head-to-head. This maps almost exactly onto the cross-pair problem: every
FX pair's move is a "match" between two currencies, and a recursive rating gives
per-currency strength. This is the natural implementation of the s7 currency
factor idea. Needs adapting from win/loss to a continuous margin (the size of
the move matters, not just its sign) — the AFL margin models are the reference
for that.

**9.2 Isotonic calibration (from PL-AFL-Module `models/calibration.py`).**
Probably the highest-value single import. Our Kelly sizer consumes RAW predicted
magnitudes (`mu/sigma^2`), and we have direct evidence they are miscalibrated:
at 1d horizon ridge predicted edges 2.09x breakeven while delivering NEGATIVE
gross Sharpe — confidently wrong. Isotonic regression maps raw model output onto
empirically-observed outcomes without assuming a functional form. Calibrating
before sizing should fix systematically oversized positions.

**9.3 Ensemble over feature subsets with disagreement (AFL `models/ensemble.py`).**
Trains N models on different feature subsets and reports mean plus spread
(`pred_margin_std`, p10/p90, `ensemble_agreement_pct`). Two wins for us:
it satisfies the s4.6 uncertainty requirement per-prediction rather than only
in aggregate, AND model disagreement is a natural "do not trade" filter. Since
costs are our binding constraint, trading only when models agree attacks the
problem from the right side.

**9.4 Paper-trading ledger (Tennis `model/paper_trading.py`, `paper_ledger.py`).**
Existing structure for the s4.7 paper-trading requirement. Reuse rather than
rebuild when we get there.

**9.5 THE TRAP — feature sweeps (Tennis `model/feature_sweep.py`).**
Randomised feature-group search that reweights toward top performers. Powerful,
and exactly the wrong tool for us right now. We have already searched ~50
configurations; Bonferroni alpha is 0.001. A sweep evaluates hundreds more and
would almost certainly surface something that looks significant and is not.
If used at all it must run ONLY on research data, with the configuration count
tracked and the correction applied — and the winner still has to clear s8 on
the sealed holdout.

**On model complexity:** we are currently using only ridge regression plus
trivial baselines. That was deliberate (baselines first), and the evidence still
says the bottleneck is signal, not model class — ridge already achieves a
respectable IC of ~0.034; it fails on costs and stability, not on capacity. A
GBM on the same features would very likely fail the same way. The imports above
are therefore about calibration, uncertainty and NEW information (currency
strength), not about a bigger model.

**Relevant literature (recalled, NOT verified — worth a search before relying
on any specific figure):** Lustig, Roussanov & Verdelhan on common risk factors
in currency markets (the "dollar factor" and carry factor); Menkhoff, Sarno,
Schmeling & Schrimpf on carry trades, FX volatility, and currency momentum;
Asness, Moskowitz & Pedersen "Value and Momentum Everywhere" which includes
currencies. **Important caveat:** these document returns on broad portfolios
often including emerging/high-yield currencies, frequently gross of realistic
retail costs, and the carry factor notably crashed in 2008. "Documented in the
literature" does not imply "survives retail costs on G10 majors" — which is
precisely the constraint that has killed everything we have tested.

---

### 2026-07-19 (session 7) — Cross-pair build started

Enabled G10 majors in config and `INSTRUMENT_MAP`: GBPUSD, USDJPY, AUDUSD,
USDCHF, USDCAD, NZDUSD (plus existing EURUSD). Added `PAIR_LEGS` mapping each
pair to its (base, quote) currency so a pair return can be attributed to two
currencies rather than one instrument.

Pulling **1h** bars 2015-2026 for the six new pairs — 1h is ample for daily/4h
factor work and ~12x faster than M15. EURUSD retains full M15 history.

**Data validated:** all six pairs 2015-2026, ~71,700 1h bars each, zero errors.
Price ranges are historically genuine — USDCHF's 0.761 low is the Jan-2015 SNB
de-peg, GBPUSD's 1.04 the 2022 mini-budget crash, USDJPY's 162.8 the 2024 peak.

**Built `src/features/currency_strength.py`.** Solves `r_pair = s_base - s_quote`
across the basket by least squares with a sum(s)=0 identification constraint,
rather than a sequential Elo update — we observe the exact margin, so the
algebra is available where tennis only has win/loss. The Elo idea survives as
the rating layer (EWMA decay == K-factor). 121 tests passing.

**Decomposition verified exactly:** `|s_EUR - s_USD - r_EURUSD| = 2.08e-17`,
machine precision. Cumulative 2015-2022 strengths are economically coherent:
CHF +19.0% (safe haven), USD +11.5% (dollar era), GBP **-13.7%** (Brexit),
commodity currencies AUD/NZD/CAD all negative. The factors are measuring
something real.

**A headline number that did not survive scrutiny.** `f_ccy_dispersion_*`
showed IC ~0.05, the strongest in the project. Year-by-year it is
-0.079, -0.041, -0.015, **+0.029, +0.028**, -0.122, +0.015, -0.015 — only 62%
sign agreement, with the pooled figure driven by 2015 (SNB) and 2020 (COVID).
Cause: dispersion is UNSIGNED, so it cannot distinguish "USD dominating" from
"EUR dominating"; it merely coincided with dollar strength in crisis years.
Rejected.

**A rolling z-score variant did not help.** Motivated by a real diagnostic (raw
`f_ccy_diff_5` averaged IC -0.055 within years but far less pooled, the
signature of level drift), but `f_ccy_diff_z_5` fell to 50% sign agreement.
Kept in the module, not used. Caveat noted: adding the z-score columns extended
the warm-up and dropped ~240 early rows, so before/after IC figures are not
quite the same sample.

**The actual result — first stable signal in the project, but below the bar.**
`f_ccy_diff_5` (EUR-minus-USD strength vs the basket, 5-bar halflife) has 88%
sign agreement and a per-year mean IC of **-0.055**, roughly 3x the ~0.02
required at daily horizon. Negative sign = short-term reversal: when EUR has
strengthened against USD relative to the basket, EURUSD tends to fall back.

Backtested across four splitter settings (criterion s8.3), research data only:

| split | ccy_reversion | ridge on all 20 strength features |
|---|---|---|
| 500/150 | **0.210** | 0.364 |
| 600/200 | **0.180** | 0.191 |
| 750/250 | **0.273** | -0.198 |
| 900/300 | **0.288** | -0.526 |

The single-feature reversion model is **stable** — 0.18 to 0.29 across every
split, turnover ~85/yr, no degeneracy flags. That is the first time anything in
this project has held its shape when the splitter moved. Ridge on all 20
features swings from +0.36 to -0.53: classic overfitting, and further evidence
that capacity is not the constraint.

**Verdict against s8: 3 of 4 criteria met.**
- (1) Net Sharpe > 0.5 — **FAILED**, ~0.24. All CIs straddle zero.
- (2) Turnover > 20/yr — passed (~85).
- (3) Stable across splitters — passed, for the first time.
- (4) No degeneracy flags — passed.

Not a pass. Do NOT pre-register this against the holdout yet — spending a
holdout look on a 0.24 candidate wastes the one clean test we have.

**Next (principled improvements, NOT more search — config count is now ~60):**
- Apply isotonic calibration (s9.2) before Kelly sizing. Still unapplied, and
  it is the highest-prior fix: sizing consumes raw magnitudes we know are
  unreliable.
- Ensemble with a disagreement filter (s9.3) — trade only on agreement, which
  attacks the cost side rather than hunting more signal.
- Combine `f_ccy_diff_5` with the price/carry features in ONE model rather than
  testing families separately.
- Resist the feature sweep (s9.5). At ~60 configurations the correction is
  already punishing.

### 2026-07-19 (session 8) — Both AFL imports built; both made things WORSE

Built `src/models/calibration.py` (isotonic return calibration, chronological
held-out calibration slice) and `src/models/ensemble.py` (feature-family
ensemble, direction-agreement filter), wired `min_agreement` into the engine.
139 tests passing, including that a skill-less model collapses toward a
constant under calibration and therefore stops trading.

**Result on the daily EURUSD cross-pair strategy, research data, 3 splitters:**

| variant | mean net Sharpe | std across splits |
|---|---|---|
| **A. base reversion (`f_ccy_diff_5`)** | **+0.257** | **0.041** |
| B. + isotonic calibration | -0.593 | 0.194 |
| C. family ensemble (ridge members) | -0.391 | 0.205 |
| D. ensemble + agreement filter | +0.022 | 0.421 |
| E. ensemble + calibration + agreement | -0.562 | 0.255 |

**The simplest model is still the best, and by far the most stable** (std 0.041
against 0.19-0.42 for everything else). Variant D produced a 0.506 at one
splitter and -0.26/-0.18 at the others — the signature of noise, not signal.

**I predicted calibration would be "probably the highest-value single import".
That was wrong, and the reason is instructive.** The AFL setting is thousands of
matches, a GBM with genuinely non-linear miscalibration, and a large calibration
sample. Ours is ~2,400 daily bars, ONE weak feature, and a near-linear
relationship. Isotonic regression is non-parametric — it spends roughly as many
degrees of freedom as it has calibration rows — so on a 150-270 row slice it
overfits, and holding that slice back also starves the base fit. The technique
was sound; the regime was wrong.

Same story for the ensemble: averaging three ridge members does not fix
overfitting when the members themselves overfit (already known — ridge on the
20 strength features swung +0.36 to -0.53 across splitters).

**Transferable lesson: our binding constraints are sample size and signal
strength, not model calibration or averaging. Techniques that ADD flexibility
should be expected to hurt here.** Import complexity only alongside more data
or a stronger signal.

Config count ~72. Bonferroni alpha 0.0007.

**Status vs s8 unchanged: 3 of 4.** Base reversion still ~0.26 against a 0.5
bar, CIs still straddle zero. Nothing here justifies a holdout look.

**Next — the honest options, in order of expected value:**
1. **More data, not more technique.** ~2,400 daily bars is the real limit. The
   same strategy on 6 more pairs (each with its own `f_ccy_diff`) would give
   ~7x the observations from ONE hypothesis, not seven — a cross-sectional
   portfolio rather than seven separate tests. This is both the standard form
   in the FX factor literature and the only move that attacks sample size.
2. Intraday (4h) cross-pair strength: ~6x more bars, though costs bite harder.
3. Accept ~0.26 is what this signal is worth and revisit the s8 bar.

### 2026-07-19 (session 9) — Cross-sectional portfolio; first candidate to meet the bar

Built `src/models/cross_sectional.py`: currency strength -> demeaned
cross-sectional scores -> per-currency weights -> minimum-norm pair trades.
152 tests passing. Repo pushed to github.com/ZHCooke/Predicto-Forex.

**Terminology bug caught by a test.** I conflated two meanings of
"dollar-neutral": the equity sense (net notional cancels) and the FX sense
(zero USD exposure). Demeaning gives the former, NOT the latter — USD is scored
like any other currency, so reversion shorts it when it has run up. That is a
deliberate view. A test asserting zero USD exposure would have been asserting
the strategy has no opinion. Diagnostics now report both separately.

**Book behaves as designed:** replication residual 0.0, net exposure 0.0,
netting ratio 0.873 (netting saves ~13% of gross trading), mean |USD| exposure
0.127.

**I OVERSTATED the sample-size argument and should correct it.** I said seven
pairs gives "~7x the observations". For estimating a portfolio Sharpe it does
not: the seven pairs collapse into ONE daily portfolio return, so the time
series is still ~2,400 points. What we actually gain is DIVERSIFICATION —
several partially-independent bets per day lowers portfolio vol for the same
edge. Real, and it showed up (max drawdown fell to -0.07 from single-pair
levels, Sharpe rose), but it is not extra statistical power along the time axis,
and the CI barely tightened.

**Halflife sweep (research data, costs per-pair from config):**

| halflife | gross Sharpe | net Sharpe | turnover | 95% CI |
|---|---|---|---|---|
| 20 | 0.498 | 0.387 | 61.2 | [-0.198, 0.973] |
| 60 | 0.516 | 0.449 | 37.6 | [-0.126, 1.015] |
| 90 | 0.550 | 0.494 | 31.9 | [-0.084, 1.050] |
| **120** | **0.573** | **0.524** | **28.7** | **[-0.079, 1.104]** |

Monotone in halflife, and GROSS Sharpe rises too — so this is not purely a
cost-reduction artifact; slower signals genuinely look better here.

**A structural advantage over everything tested before: the strategy has NO
FITTED PARAMETERS.** It is purely rules-based (decompose, demean, invert,
normalise). There is no train/test fitting and therefore no model overfitting —
which is why it holds its shape where ridge swung from +0.36 to -0.53.

**Verdict vs s8 — the first candidate to pass all four, WITH CAVEATS:**
- (1) Net Sharpe > 0.5 — PASSES at halflife 120 (0.524).
- (2) Turnover > 20/yr — passes (28.7).
- (3) Stable — passes; monotone and smooth across halflives, no fitted params.
- (4) No degeneracy — passes.

**Why this is still not a result:**
1. **The CI includes zero** ([-0.079, 1.104]). The s8 criteria never required a
   CI excluding zero — that is a GAP IN THE BAR WE SET, visible only now that
   something has crossed it. At Bonferroni alpha 0.0006 (~84 configs) this is
   nowhere near significant.
2. **Halflife 120 was chosen as the argmax of a sweep.** Selecting the best of
   six and reporting it is exactly the error the bar exists to prevent.
3. **Effective sample is ~20 independent observations.** A 120-day halflife
   means the signal turns over roughly twenty times in eight years. A Sharpe of
   0.52 on ~20 effective bets is weak evidence regardless of the point estimate.

**Recommendation: this is the first candidate worth spending the holdout on.**
Pre-register a slow cross-sectional reversion book (halflife 90-120) with a
predicted net Sharpe, then take the single permitted look. NOT yet done —
spending the holdout is irreversible and is the owner's call.

**SUPERSEDED BY SESSION 10 — do not act on the above. Per-prediction scoring
shows the halflife-120 candidate has no directional skill; the 0.524 was noise.**

---

## 10. Per-Prediction Scoring — the power problem, solved

`src/backtest/scoring.py`. Owner's suggestion, and it broke a genuine deadlock.

**The problem it solves.** Sharpe compresses a year into one number, so 8 years
gives ~8 effective observations and needs ~17 years to prove S = 0.5. Every
result in this project had a CI straddling zero for that reason alone.
Per-prediction scoring grades every forecast individually: **2,499 effective
observations** (17,493 raw forecasts, blocked by date) instead of ~8.

**Design decisions, and what the AFL module actually does.**
PL-AFL-Module computes log loss / Brier / ROC-AUC per round and accumulates
them, but **never fits to them** — its models are regressions on margin, with
probabilities from calibration afterwards. We copied that deliberately.
Optimising for log loss would be optimising AGAINST profitability, because log
loss weights every observation equally whereas P&L weights by the size of the
move. A model tuned for log loss would buy accuracy on many tiny moves at the
cost of the few large ones that pay for the spread.

**Measured correction to my own claim: log loss is NOT the most powerful test.**
For a calibrated 54% forecaster over 2,000 observations:
accuracy z = 3.58, log-loss z = **1.79** — half the power. A small calibrated
edge barely moves the loss (edge ~ 2*(p-0.5)^2) while outcome noise stays large.
Log loss earns its place by grading CALIBRATION, not by detecting direction.
`signed_return_test` is therefore the primary test: it weights by move size,
which is the economically meaningful quantity.

**A real bug caught here — a fabricated significance.** The degenerate-variance
guard used `se > 0`. A constant loss series has floating-point std ~1e-17, which
passes, producing **t = 6.5e16, p = 0.0** from data containing no information.
Now guarded with a threshold relative to the magnitude tested. This is exactly
the class of error that manufactures false discoveries.

**Blocking matters and is easy to get wrong.** Correlated same-day forecasts
must count as one unit. Note the subtlety found while testing: correlating the
DIRECTION called does nothing — the losses only correlate if CORRECTNESS is
shared across the day, which is what a common-factor day looks like in FX.

### Result: the halflife-120 candidate has no directional skill

Cross-sectional reversion, research data, all seven pairs scored per-pair-per-day:

| halflife | accuracy | log-loss edge | signed-return t | p |
|---|---|---|---|---|
| 3 | 0.5077 | -0.0284 | 1.557 | 0.120 |
| **5** | **0.5075** | -0.0169 | **1.580** | **0.114** |
| 20 | 0.4961 | -0.0058 | 0.654 | 0.513 |
| 60 | 0.4949 | -0.0036 | 0.476 | 0.634 |
| **120** | **0.4967** | -0.0031 | 1.042 | 0.298 |

**Nothing reaches significance at any halflife.** Best is p = 0.114 at halflife
5, against a Bonferroni alpha of 0.0006 after ~84 configurations.

**The decisive detail: halflife 120 — the setting that produced net Sharpe
0.524 and "passed" all four s8 criteria — has directional accuracy of 49.67%,
BELOW a coin flip.** Its apparent profitability was not predictive skill.
Short halflives (3-5) show the only hint of genuine direction (50.8%), and even
that is not significant.

**Log-loss edge is negative at every halflife**, i.e. our probability forecasts
are systematically OVERCONFIDENT. That independently confirms the session-8
hypothesis about miscalibrated magnitudes feeding Kelly — now measured rather
than assumed.

### Consequences for the two open decisions

1. **Do NOT spend the holdout on this candidate.** It fails a far more powerful
   test on research data we are still free to use. Spending the one clean look
   on it would waste it.
2. **The "17 years" objection is retired.** We now have a significance test with
   ~2,499 effective observations. A future candidate can be adjudicated on
   research data BEFORE the holdout is touched — which is the correct order and
   was not previously possible.

**New gate, to sit ahead of s8:** any candidate must clear `signed_return_test`
on research data (p below the Bonferroni-corrected threshold) BEFORE its Sharpe
is taken seriously or the holdout is opened. Sharpe is now the second filter,
not the first.

---

### 2026-07-19 (session 11) — Full feature screen; ONE survivor

`src/analysis/feature_screen.py`. Every feature family (price, carry, strength)
at daily and 4h, signed-return test, pooled across pairs, blocked by date,
research data only. 82 tests, Bonferroni over 82 + 84 prior configs -> alpha
0.000301.

**The corpus is empty.** Only **1 of 82** features passed even an UNCORRECTED
p < 0.05, where 4.1 would be expected by chance. Fewer nominal hits than chance
produces is strong evidence there is essentially no directional information in
anything we built: momentum, mean-reversion, RSI, ATR, carry, rates changes,
currency strength, dispersion, dollar factor. Ten sessions of feature work,
now rigorously null rather than inconclusively null.

`f_ccy_diff_5` (the session-9 candidate) came in at t = -1.41, p = 0.16.
Confirms the session-10 verdict.

**The single survivor: `f_hour_cos` at 4h, t = 4.34, p < 0.0001.** Intraday
seasonality, not a model feature.

**Two corrections while chasing it down.**
1. Not a rollover artifact — survives excluding the 20:00 bar (t = 3.92).
2. My first read blamed the 16:00 bar. WRONG: four pairs are XXXUSD and three
   are USDXXX, so pooling raw pair returns mixes directions. Re-expressed in
   USD terms, 16:00 is insignificant (t = -0.88) and the real effect is at
   **08:00 UTC, the London open (t = 4.92)** — USD strengthens systematically.

**Why only EURUSD can harvest it.** The effect is positive on all seven pairs
gross, but only EURUSD's spread is tight enough:

| pair | gross (pips) | t | round-trip cost | net |
|---|---|---|---|---|
| **EURUSD** | **2.186** | **4.01** | 0.80 | **+1.386** |
| USDCAD | 1.016 | 2.15 | 1.70 | -0.684 |
| USDJPY | 1.010 | 2.13 | 1.10 | -0.090 |
| others | 0.86-1.10 | 1.2-1.7 | 1.3-1.9 | negative |

### The candidate: short EURUSD in the 08:00-12:00 UTC bar

A CALENDAR RULE with **no fitted parameters** — nothing to overfit but the hour.

| metric | value |
|---|---|
| Gross Sharpe | 1.376 |
| **Net Sharpe** | **0.873** |
| **Net Sharpe 95% CI** | **[0.196, 1.588]** — EXCLUDES ZERO |
| Annual return | 3.4% (unlevered) |
| Max drawdown | -7.6% |
| Trades | 247/yr |
| Years positive (gross) | **8 / 8** |

**First CI in this project to exclude zero.** Clears every gate: the corrected
signed-return screen, Sharpe > 0.5, turnover > 20, stability, no degeneracy.

**Honest caveats before anyone gets excited:**
1. **Selection.** The hour was read off a six-row table and EURUSD chosen
   because it is the only affordable pair — an implicit 6 x 7 = 42 choices on
   top of the 82-feature screen. Bonferroni over 42 gives alpha 0.0012;
   t = 4.01 implies p ~ 0.00006, so it still clears, but the margin is smaller
   than the headline p-value suggests.
2. **DST is unhandled.** London opens at 07:00 UTC in summer and 08:00 in
   winter; our fixed 4h buckets ignore this, so the estimate mixes two regimes.
   The effect may be stronger with DST-aware bucketing — or the current number
   may be flattered by it. Must be checked.
3. **This is a KNOWN effect.** Intraday FX seasonality around the London open
   is documented. Reassuring for validity, but it means we are rediscovering,
   not discovering, and known effects decay as they are crowded.
4. **3.4% unlevered** is modest in absolute terms.
5. **Not yet tested on the holdout.**

**Recommendation: THIS is the candidate to spend the holdout on.** Pre-register
"short EURUSD 08:00-12:00 UTC" with a predicted net Sharpe (shrunk for
selection — suggest 0.5, not 0.873), fix the DST handling first, then take the
single permitted look.

---

### 2026-07-19 (session 12) — Structural features, regime tests, and a data-quality bug

Built `src/features/structural.py` (DST-aware sessions, month/quarter-end,
expiry, day-of-week) and `src/analysis/regime.py` (trailing-quantile regimes,
conditional and interaction tests). 186 tests passing.

**Datasets confirmed reachable (all free):**
- FRED `VIXCLS` (1990-), `T10Y2Y` yield curve (1976-) — full history.
- FRED `SP500` only goes back to 2016 and `BAMLH0A0HYM2` to 2023 — both
  truncated by FRED, unusable for our 2015 start. VIX covers the equity-stress
  regime anyway, being derived from SPX options.
- **CFTC COT positioning IS accessible** — the plain URL 403s, but it works
  with a `User-Agent` header; yearly history zips at
  `cftc.gov/files/dea/history/fut_fin_txt_<year>.zip`. Not yet integrated.

**Structural features: nothing survives correction.** 19 tests, cumulative
alpha 0.00027. Best was `f_sess_london_open` (t = 2.68) and `f_is_year_end`
(t = -2.65); 2 nominal hits where 1.0 is expected by chance. Month-end,
quarter-end and option-expiry flags are all flat.

**Regime conditioning: the hypothesis is REFUTED, not merely unproven.**
0 of 7 interactions significant (0.4 expected by chance) for momentum,
z-score, RSI, currency strength, dispersion and 1-bar return across
low/high realised-vol regimes. Signals are not cancelling across regimes —
they simply are not there. This closes the methodological gap flagged in
session 11: it was a real gap, and it was not the explanation.

### A genuine data-quality bug: BID-ONLY PRICES CREATE FAKE SIGNAL

The 1h scan showed enormous t-statistics clustered in the rollover window
(+8.6 at 20:00 London, -8.3 at 22:00, +5.1 at 23:00) with alternating signs —
the signature of a quote dislocation that mean-reverts, not a tradeable effect.

We have used `OFFER_SIDE_BID` throughout. Re-running EURUSD 2019-2022 on MID
prices (bid/ask average):

| London hour | bid t | mid t | effect |
|---|---|---|---|
| 20 | **7.96** | **1.57** | collapses ~80% |
| 21 | -4.63 | -1.49 | collapses |
| 22 | -6.49 | -3.95 | halves |
| **8 (London open)** | **1.976** | **1.964** | **unchanged** |

**The rollover "signal" was almost entirely a bid-side artifact.** At thin
liquidity the bid gaps and recovers while mid barely moves, manufacturing a
down-up pattern that is not tradeable. Had we screened at 1h earlier and not
checked, this would have been a confident false discovery.

**The London-open effect is NOT an artifact** — it is identical on bid and mid,
which is exactly what a real flow-driven effect should look like.

**ACTION REQUIRED: switch the pipeline to mid prices.** `fetch_dukascopy`
should pull both sides and store mid (or store both). Everything touching
illiquid hours is suspect until this is done. Liquid-hours results (including
the London-open candidate) are unaffected.

### Status of the London-open candidate after DST correction

Tested properly at 1h resolution with DST-aware local-time flags:
- 08:00-10:00 London -> forward **4h** return: t = 3.683, p = 0.00023, +1.09 pips
- 08:00-10:00 London -> forward **1h** return: t = 0.877, p = 0.38

So it is a session-length phenomenon, not an hourly one — consistent with
flow accumulating over the London morning. p = 0.00023 sits just inside the
cumulative Bonferroni alpha of 0.00027, i.e. it survives, but marginally.

Note the DST-aware 4h test was WEAKER (t = 2.68) than the fixed-UTC version
(t = 4.92). That is a resolution artifact, not a contradiction: 4h bars sit on
fixed UTC boundaries, so a DST-shifted local session cannot be represented
cleanly on that grid. The 1h test above is the correct one.

**Next:**
1. Switch to mid prices — data integrity, blocks everything else.
2. Integrate COT positioning (the one genuinely new data source available free).
3. Re-verify the London-open candidate on mid prices, then pre-register.

---

### 2026-07-19 (session 13) — Mid prices, COT, and the overlapping-returns trap

200 tests passing.

**1. Mid prices.** `fetch_dukascopy` now supports `offer_side="mid"`, fetching
both sides and averaging, plus a measured `spread` column (better than the
hourly-median profile we were using). Mid partitions live under
`<timeframe>_mid/` so the two price conventions can never be silently blended.
Documented approximation: mid-high is the average of bid-high and ask-high,
which is exact only if spread is constant intrabar — irrelevant to any result
here, since every return is computed from CLOSES, which are exact.

**2. CFTC COT integrated** (`src/ingest/fetch_cot.py`). Weekly speculative
positioning for all seven currencies, 2014-2026, 654 observations each.

Two data bugs found and fixed:
- **Prefix matching was wrong.** `startswith("EURO FX")` also matched
  "EURO FX/BRITISH POUND XRATE" and "EURO FX/JAPANESE YEN XRATE" — different
  instruments entirely — silently doubling EUR to 1407 rows. Now exact
  full-contract-name matching.
- **NZD appeared to end in 2022.** CFTC renamed the contract from
  "NEW ZEALAND DOLLAR" to "NZ DOLLAR"; both names are now mapped.

Publication lag handled: the report snapshots Tuesday but publishes Friday
15:30 ET, so `align_to_bars` releases it from the following Monday
(`release_lag_days=6`), and lags under 4 days raise.

### THE MAIN FINDING: overlapping forward returns fabricate significance

COT positioning at a 1-month horizon looked like the best result in the project:
`f_cot_diff_lev` t = **-3.75**, p = 0.0002, with **11 of 39 features nominally
significant where 2 are expected** — a coherent contrarian cluster, exactly the
documented hypothesis. It cleared the cumulative Bonferroni threshold and was
headed for the holdout.

It is an artifact. A 20-day forward return sampled daily overlaps 95% with its
neighbours; blocking by date fixes CROSS-SECTIONAL correlation but does nothing
for SERIAL overlap.

| sampling | t | p | n_eff |
|---|---|---|---|
| blocked by date (overlapping) | **-3.75** | 0.0002 | 2382 |
| blocked by month | -1.19 | 0.238 | 93 |
| **strictly non-overlapping** | **-0.74** | **0.461** | 120 |

The entire result came from counting 2,382 overlapping observations as
independent when there were about 120. **COT positioning shows nothing.**

Guarded in `scoring.py`: `signed_return_test` takes `horizon_bars` and emits an
`overlap_warning` estimating the sqrt(H) inflation, with a regression test.

### The London-open candidate SURVIVES the same test

Re-checked, since it also used a multi-bar (4h) forward return:

| sampling | t | p | n_eff |
|---|---|---|---|
| blocked by bar (overlapping) | 3.683 | 0.00023 | 4152 |
| blocked by day | 2.945 | 0.00327 | 2077 |
| **one bar/day, non-overlapping** | **3.282** | 0.00105 | 2077 |
| **EURUSD alone, non-overlapping** | **5.872** | **<0.00001** | 2076 |

It degrades modestly rather than collapsing — the signature of a real effect.
EURUSD alone gives **+2.856 pips gross against ~0.8 pips cost** on 2,076
strictly independent daily observations.

Note this also resolves the session-12 confusion about DST making things worse:
measured at 1h resolution the DST-aware version is STRONGER (2.856 vs 2.186
pips). The earlier weakening was purely the 4h fixed-UTC grid being unable to
represent a DST-shifted session.

**The candidate has now survived four independent attacks:** mid-price check,
overlap correction, Bonferroni over ~180 configurations, and 8/8 positive years.

**Next:** finish the mid-price pull (5/7 done), re-verify the candidate on mid
data, then pre-register and take the single holdout look.

---

## 14. Session Index

Quick map of where each finding lives. Sessions are dated entries in s6 above
plus the numbered sections.

| # | What happened | Key outcome |
|---|---|---|
| 1 | Repo scaffolded, invariants mutation-tested | Pipeline validated in both directions (oracle profits, stale signal loses) |
| 2 | Multi-year data pulled | Partition idempotency bug found — existence-only check nearly left an 11-month hole |
| 3 | Costs diagnosed | Required IC is 0.19 at M15 vs ~0.02 daily. Horizon, not model, was the problem |
| 4 | FRED rates wired in | Rates do not predict spot direction. Carry mis-specified as a cost, then fixed |
| 5 | Holdout sealed at 2023-01-01 | h001 pre-registered and evaluated: INCONCLUSIVE (traded 0.89% of bars) |
| 6 | Cost-side bugs fixed | `min_edge` was ~7x too strict; session filter cost more than it saved. Earlier results were biased pessimistic |
| 7 | Cross-pair currency strength built | Decomposition exact to 2e-17; economically coherent (GBP weakest post-Brexit, CHF strongest) |
| 8 | AFL imports (calibration, ensemble) | Both made results WORSE. Constraint is sample size, not model sophistication |
| 9 | Cross-sectional portfolio | Net Sharpe 0.524 at halflife 120 — passed all four criteria, later shown to be noise |
| 10 | Per-prediction scoring built | Sharpe cannot resolve anything here (~17 years needed). The halflife-120 candidate had 49.67% accuracy — below a coin flip |
| 11 | Full 82-feature screen | **The corpus is empty**: 1 nominal hit where chance gives 4.1. One survivor: London-open seasonality |
| 12 | Structural + regime tests | Regime conditioning refuted (0/7). **Bid-only prices fabricate signal at rollover** |
| 13 | Mid prices, COT, overlap trap | **COT's t = -3.75 was overlap artifact** (-0.74 corrected). London-open candidate survived |
| 14 | Holdout spent on h002 | Effect REAL out of sample but half-size: 1.03 pips, net Sharpe 0.197. Too small at 0.80 cost. Framework rubber-stamped it CONFIRMED anyway (2nd flaw) |
| 15 | History extended to 2005; tick execution study | 21.5y, p=1e-9, no decay, replicated on fresh 2005-2014 data. Cost measured at 0.345 pips (not 0.80) — clears 0.5 bar. Plain-English PDF written |

---

## 15. If You Are Picking This Up Cold

1. Read **s0** — current state, unfinished pull, the candidate.
2. Read **s0.5** — the nine traps. Each one produced a convincing false result
   before it was caught, and most are not obvious in advance.
3. Run `pytest -q` (expect 200 passing) to confirm the environment.
4. Finish the pull in **s0.1** if it is still incomplete.
5. Do NOT touch the holdout without a pre-registration (`src/backtest/holdout.py`
   enforces this and logs every access).
6. Any new idea gets tested with `signed_return_test` on RESEARCH data first,
   with overlap handled, before its Sharpe means anything.

The honest summary after thirteen sessions: **the pipeline is solid and the
alpha search has failed**, with one structural candidate still unspent. That is
the expected outcome for retail-accessible major-pair FX, and it is now
established rigorously rather than vaguely — which is the actual deliverable so
far.

---

### 2026-07-19 (session 14) — Holdout spent on h002. Effect is real, and too small.

Verified the candidate on mid data (t 5.872 -> 5.912, indistinguishable), ran
five robustness diagnostics (all supportive — see s0.1), pre-registered h002 and
committed it to git BEFORE looking, then took the single permitted holdout look.

**Result: gross +1.025 pips (t = 1.707, p = 0.088), net Sharpe 0.197, annual
return 0.52%.** See s0.3a for the full table.

**The three things worth carrying forward:**

1. **Replication in shape, not magnitude.** Days-positive was 54.9% vs 55.7% in
   research and median pips 1.89 vs 2.41 — the effect is clearly there. But the
   MEAN edge halved, which is what happens when a selected estimate meets fresh
   data. Direction replicated; size did not.

2. **Costs, one final time.** Gross 1.025 pips against a 0.80 pip round trip
   leaves 0.225. The entire eight-session cost investigation is vindicated: at
   retail spreads, an effect has to be several times larger than this to matter.
   This is the same wall every strategy in this project has hit.

3. **A framework flaw found by being on the receiving end of it.** The
   pre-registration returned CONFIRMED for a strategy that fails the success
   criteria, because 0.197 sits inside 0.50 ± 0.40. Session 5 taught us that
   confirming a DEGENERATE result is worse than no framework; session 14 adds
   that confirming an ECONOMICALLY IRRELEVANT one is the same failure wearing a
   different hat. Fix: `evaluate_against_prereg` should require the s8 bar, and
   tolerances should be tight enough that CONFIRMED implies actionable.

**Where this leaves the project.**

The holdout is spent. There is no clean out-of-sample data left for 2015-2026,
and per s4.7 the only honest validation remaining is FORWARD paper trading on
data that does not yet exist.

The strategy search has now returned a complete answer rather than an ambiguous
one: price-derived features are empty (s11), regime conditioning is refuted
(s12), positioning is an overlap artifact (s13), and the one genuine structural
effect is real but roughly 4x too small to clear retail costs (s14). That is not
a failure to find the answer; it is the answer.

**If continuing, the only directions with a real prior:**
- **Lower costs**, not better signal. The effect clears at ~0.2 pip round trip
  (institutional/ECN pricing with rebates), and is dead at 0.8. Everything we
  have found is cost-constrained rather than signal-constrained.
- **Forward paper trading** the London-open rule to gather genuinely clean data,
  accepting it is currently ~0.5%/yr.
- **A different instrument class** where retail costs are proportionally lower
  relative to volatility.
- Otherwise: treat the pipeline as the deliverable and stop.

### 2026-07-19 (session 15) — Extended to 2005, measured real execution cost, wrote the explainer

Three pieces of work, and the conclusion from session 14 (h002 "too small to
trade") is now REVERSED — not because the edge grew, but because the cost
assumption was wrong. Full detail is in s0.0b-e at the top; summary here.

**1. History extended to 2005.** Pulled 2005-2014 mid data for all seven majors
(~144k bars each, 0 validation errors). 21.5 years total, which for the first
time crosses the ~17.3 years needed to resolve a Sharpe of 0.5 (s0.5 trap 3).

The London-open effect over the full history: **1.858 pips, t = 6.12,
p = 1e-9**, 20/22 years positive, decay trend p = 0.63 (none). Crucially it
REPLICATED on the 2005-2014 decade that played no part in finding it
(1.402 pips, t = 2.98, p = 0.0029) — the first genuinely independent replication
in the project. True selection-free size is 1.314 pips (research overstated by
2.2x). It is a USD phenomenon: positive on 7/7 pairs, significant on 3 (EURUSD,
USDJPY, USDCHF); EURUSD alone is the best book, since diversification cannot
rescue pairs whose spread exceeds their edge.

**2. Tick-level execution study** (`src/analysis/execution.py`). Reconstructed
actual fills from real bid/ask ticks (~80k/day) on 30 sampled days.

- **Bar-labelling bug found:** Dukascopy bars are START-labelled, so the trade
  is enter 09:00 London / exit 13:00, not 08:00/12:00. No statistical result
  changes; h002 pre-reg was written precisely enough to be unaffected; any LIVE
  implementation must use 09:00. See s0.0d.
- **Measured round-trip cost: 0.345 pips**, not the 0.80 assumed for eight
  sessions — 2.3x too harsh. At 0.345, the selection-free strategy nets 0.969
  pips, **Sharpe 0.636, CI [0.212, 1.064] — first out-of-sample result in the
  project with a CI excluding zero.**

**The whole project now reduces to ONE unmeasured number: broker slippage,
budget ~0.18 pips/leg.** Spread is measured, edge is measured over 21 years,
slippage is not and cannot be measured from bar data — it needs live micro-lot
execution (a demo account fills at the quoted price and flatters exactly this
number). See s0.0e.

**3. Plain-English explainer** (`docs/`). `docs/project_explainer.html` (source)
renders to `docs/Predicto-Forex-Explained.pdf` (8 pages, headless Chrome — see
`docs/README.md`) for a reader with no forex background. Deliberately documents
the failures and the two exciting-then-destroyed results, because the pattern of
caught mistakes is the most transferable content.

**Status: the answer flipped from session 14.** The effect is real, replicated,
undecayed over 21 years, and at the MEASURED (not assumed) cost it clears the
s8 bar out of sample. The remaining question is no longer "is there signal" —
21 years and p = 1e-9 settled that — but whether a real broker's slippage stays
under ~0.18 pips/leg. That is a live-execution question, and the next step is
instrumented micro-lot trading, not more research.

---

### 2026-07-30 (session 16) — Cross-asset session-transmission battery (h003): NO tradeable lead

Owner asked whether cross-asset datasets (equities, gold, oil, copper, gas, VIX,
DXY) could add predictive power. Most of the requested list was already ruled out
(rates s4, COT s13, vol/regime s12, ATR s11), so the only genuinely new,
defensible idea was a **session-transmission LEAD**: does a cross-asset move
during one region's session predict the NEXT open of a currency that was closed
while it moved? (The rest is contemporaneous co-movement — untradeable.)

**Built:** `src/features/cross_asset.py` (session-window returns + lookahead-proof
UTC pairing), `src/analysis/session_lead.py` (the battery + BH-FDR),
`tests/test_session_lead.py`. Pulled 1h mid CFDs from the SAME Dukascopy feed
(SP500/NASDAQ/DOW 2012+, GOLD 2003+, WTI/BRENT/COPPER/NATGAS ~2011-12+). VIX
(data only from 2022), DXY (2017, near-circular with the USD strength factor) and
bond CFDs (2018) excluded on data-depth grounds. 205 tests passing.

**Pre-registered 17 cells (15 tradeable + 2 contemporaneous controls) and
committed to git BEFORE running** (`preregistrations/h003_*.json`, commit
26523e4). Primary stat: OLS slope with intercept (absorbs the h002 London-open
drift), one-sided in the pre-registered direction; cross-check `signed_return_test`.

**RESULT: no tradeable lead. Four cells pass the OLS-slope FDR gate (A2/A3/A5
SP500/NASDAQ->AUD/NZD, B3 SP500->GBP) but ALL FOUR are artifacts:**
- `signed_return_test` (the primary ECONOMIC test, s10) is ~0 for all four
  (t = 0.4/-0.2/0.9/1.0).
- Trimming 5% of predictor tails collapses the correlation from ~0.07 to ~0.00.
- Directional **hit rate is BELOW 50%** (0.485-0.495).
- i.e. the whole signal is a handful of big risk-on/off days co-moving across
  the overnight gap — tail co-movement, not a directional edge. This is the s0.1
  outlier diagnostic in REVERSE: a real effect strengthens under trimming; these
  vanish.

**The contemporaneous controls confirm the mechanism.** GOLD vs EURUSD in the
SAME window: corr 0.40, t = 29.5. SP500 vs EURUSD same window: t = 4.5. The
cross-asset link is strong CO-MOVEMENT and ~0 as a LEAD — exactly the
contemporaneous-not-predictive thesis. This is the definitive answer to "do these
datasets help": they explain the present, they do not predict the next session.

**One hypothesis-generating observation (NOT a result — opposite to
pre-registration, so reporting it as confirmed would be post-hoc sign-flipping).**
The strongest *statistical* lead is a REVERSAL: US-afternoon risk-on ->
JPY/CHF (havens) STRENGTHEN at the next open (A1 SP500->USDJPY t = -3.72, A4
NASDAQ->USDJPY -3.61, B1 SP500->USDCHF -2.08). Unlike the survivors these HOLD
under trimming (corr -0.07 -> -0.07) and have a coherent ~55% hit rate in the
reversed direction. Mechanistically plausible (overnight profit-taking / partial
mean-reversion of the risk move). If pursued it must be a FRESH pre-registration
tested on untouched data (gold cells have 2003-2014; JPY/CHF cells only forward),
and it is almost certainly cost-bound at USDJPY/USDCHF spreads anyway.

**Verdict:** cross-asset series are added to the "ruled out as a LEAD" pile with
the same rigor as the rest (s0.7). The London-open rule remains the only live
candidate, and the next step is unchanged: instrumented micro-lot execution, not
more features. h003 is a research battery on (already-searched) 2015-2022 data,
so even the reversal is a candidate, not a green light — the green light still
requires forward paper trading.

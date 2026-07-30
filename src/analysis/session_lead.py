"""
session_lead.py

The cross-asset SESSION-TRANSMISSION battery (CLAUDE.md s16).

Each hypothesis asks: does the move in a cross-asset series during one region's
session predict the NEXT open of a currency that was closed while it moved? See
`src/features/cross_asset.py` for why that is the only structurally defensible
LEAD, and why lookahead and the overlap trap cannot bite here.

THE FULL BATTERY IS DECLARED IN THIS FILE AND COMMITTED TO GIT BEFORE IT IS RUN.
That is what fixes the configuration count: pre-registration converts an
open-ended search (which inflates false positives without bound, s10/s13) into a
fixed-size test whose multiplicity we can correct honestly. Adding a cell after
seeing results would void the correction.

PRIMARY STATISTIC — an OLS slope with an intercept, not a bare correlation.
    target_ret = a + b * predictor_ret + e
The intercept `a` absorbs any UNCONDITIONAL session drift, which matters because
some target windows (the London open) carry a known standalone effect (h002). We
are testing whether the predictor adds information BEYOND that constant, so the
slope `b` is the right quantity and a raw sign test would confound the two.

DIRECTION IS PRE-REGISTERED. Each cell commits to an expected sign from its
mechanism, and the reported p-value is ONE-SIDED in that direction. Committing
the sign in advance removes the free "maybe it works inverted" parameter that a
two-sided screen would leave open for a trading rule.

GATE. Benjamini-Hochberg FDR at q = 0.10 across the pre-registered battery
(survivors are a FILTER, not a verdict — they still face confirmation on the
2003-2014 gold slice where available, else forward paper trading). The
Bonferroni-over-cumulative figure is reported alongside for context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.holdout import research_only
from src.backtest.scoring import signed_return_test
from src.features.cross_asset import (
    LONDON,
    NEW_YORK,
    SYDNEY,
    TOKYO,
    Window,
    pair_contemporaneous,
    pair_lead,
    session_frame,
)

log = logging.getLogger(__name__)

# Configurations searched in sessions 1-15 (see feature_screen.PRIOR_CONFIGURATIONS
# = 84, plus the ~96 further configs and diagnostics through s15 that bring the
# project total to ~180). Carried into the Bonferroni-for-context figure so the
# cumulative bar reflects everything ever looked at, not just this battery.
PRIOR_CONFIGURATIONS = 180

# --- Windows, in each market's own local time (DST-aware) -------------------
US_PM = Window(NEW_YORK, 12, 16, "US_afternoon")          # after London has closed
LON_OPEN = Window(LONDON, 8, 12, "London_open")           # the h002 window
TOK_OPEN = Window(TOKYO, 9, 12, "Tokyo_open")
SYD_OPEN = Window(SYDNEY, 8, 12, "Sydney_open")
NY_OPEN = Window(NEW_YORK, 8, 12, "NY_open")
ASIA_CMD = Window(TOKYO, 9, 15, "Asia_commodity")         # Asian commodity session


@dataclass(frozen=True)
class Hypothesis:
    hid: str
    predictor: str
    pred_window: Window
    target: str
    tgt_window: Window
    sign: int            # +1 / -1: pre-registered expected slope direction
    group: str
    mechanism: str
    tradeable: bool = True          # False for the contemporaneous controls
    pred_weekday: int | None = None  # restrict predictor to one weekday (weekend tests)


# ---------------------------------------------------------------------------
# THE BATTERY. Frozen list — do not append after a run.
# ---------------------------------------------------------------------------
BATTERY: list[Hypothesis] = [
    # Group A — US equities -> next ASIAN open (shortest, cleanest lead: only a
    # few hours between US close and the Tokyo/Sydney open, no full session in
    # between). Risk-on = funding/haven JPY weakens (USDJPY up), risk currencies
    # AUD/NZD strengthen.
    Hypothesis("A1", "SP500", US_PM, "USDJPY", TOK_OPEN, +1, "A", "risk-on -> JPY sold as funding -> USDJPY up"),
    Hypothesis("A2", "SP500", US_PM, "AUDUSD", SYD_OPEN, +1, "A", "risk-on -> AUD (risk ccy) up"),
    Hypothesis("A3", "SP500", US_PM, "NZDUSD", SYD_OPEN, +1, "A", "risk-on -> NZD (risk ccy) up"),
    Hypothesis("A4", "NASDAQ", US_PM, "USDJPY", TOK_OPEN, +1, "A", "tech risk proxy -> JPY weak -> USDJPY up"),
    Hypothesis("A5", "NASDAQ", US_PM, "AUDUSD", SYD_OPEN, +1, "A", "tech risk proxy -> AUD up"),

    # Group B — US equities -> next EUROPEAN open (weaker prior: Asia trades in
    # between and may already absorb the move). CHF is the clean risk-sensitive
    # leg (safe haven); EUR/GBP vs USD are ambiguous, included at lower prior.
    Hypothesis("B1", "SP500", US_PM, "USDCHF", LON_OPEN, +1, "B", "risk-on -> CHF (haven) sold -> USDCHF up"),
    Hypothesis("B2", "SP500", US_PM, "EURUSD", LON_OPEN, +1, "B", "risk-on -> USD mildly soft -> EURUSD up (ambiguous)"),
    Hypothesis("B3", "SP500", US_PM, "GBPUSD", LON_OPEN, +1, "B", "risk-on -> GBP up vs USD (ambiguous)"),

    # Group C — oil -> CAD. Oil price discovery is centred on the US session;
    # CAD is the petro-currency. Oil up -> CAD strong -> USDCAD down.
    Hypothesis("C1", "WTI", US_PM, "USDCAD", NY_OPEN, -1, "C", "WTI up -> CAD strong -> USDCAD down"),
    Hypothesis("C2", "BRENT", US_PM, "USDCAD", NY_OPEN, -1, "C", "Brent up -> CAD strong -> USDCAD down"),

    # Group D — overnight commodity (Asian session) -> commodity-currency
    # European open, SAME day (Asia sets the commodity tone before Europe reacts).
    Hypothesis("D1", "COPPER", ASIA_CMD, "AUDUSD", LON_OPEN, +1, "D", "copper up in Asia -> AUD up at London open"),
    Hypothesis("D2", "COPPER", ASIA_CMD, "NZDUSD", LON_OPEN, +1, "D", "copper up in Asia -> NZD up at London open"),
    Hypothesis("D3", "GOLD", ASIA_CMD, "AUDUSD", LON_OPEN, +1, "D", "gold up in Asia -> AUD up at London open"),

    # Group E — weekend gap: Friday US afternoon -> Monday open (predictor
    # restricted to Friday; pair_lead bridges the weekend to Monday).
    Hypothesis("E1", "SP500", US_PM, "USDJPY", TOK_OPEN, +1, "E", "Fri risk -> Mon Tokyo open JPY", pred_weekday=4),
    Hypothesis("E2", "SP500", US_PM, "EURUSD", LON_OPEN, +1, "E", "Fri risk -> Mon London open EUR", pred_weekday=4),

    # Group F — CONTEMPORANEOUS CONTROLS (same window, not tradeable). If these
    # are strong while the leads above are ~0, the cross-asset link is
    # co-movement, not predictability — the whole point the study must rule on.
    Hypothesis("F1", "SP500", LON_OPEN, "EURUSD", LON_OPEN, +1, "F", "contemporaneous risk co-move", tradeable=False),
    Hypothesis("F2", "GOLD", LON_OPEN, "EURUSD", LON_OPEN, +1, "F", "contemporaneous gold/USD co-move", tradeable=False),
]


def _ols_one_sided(pred_ret: pd.Series, tgt_ret: pd.Series, sign: int) -> dict:
    """OLS slope test, one-sided in the pre-registered `sign` direction."""
    res = stats.linregress(pred_ret.values, tgt_ret.values)
    # linregress p is two-sided for H0: slope == 0. Convert to one-sided in the
    # committed direction: if the observed slope matches the pre-registered sign,
    # the one-sided p is half the two-sided; otherwise it is 1 - that.
    matches = np.sign(res.slope) == np.sign(sign)
    p_one = res.pvalue / 2 if matches else 1 - res.pvalue / 2
    return {
        "slope": float(res.slope),
        "r": float(res.rvalue),
        "t_stat": float(res.slope / res.stderr) if res.stderr else np.nan,
        "p_two_sided": float(res.pvalue),
        "p_one_sided": float(p_one),
        "sign_matches": bool(matches),
    }


def run_hypothesis(h: Hypothesis) -> dict:
    """Evaluate one cell on RESEARCH data only (2015-2022)."""
    pred = session_frame(h.predictor, h.pred_window, weekday=h.pred_weekday)
    tgt = session_frame(h.target, h.tgt_window)

    if h.tradeable:
        paired = pair_lead(pred, tgt)
        # Enforce the lead: every target must start at/after its predictor ends.
        assert (paired["gap_hours"] >= 0).all(), f"{h.hid}: lookahead — target precedes predictor"
    else:
        paired = pair_contemporaneous(pred, tgt)

    # Research window only. tgt_start is when the position would be taken, so it
    # is the correct timestamp to gate on.
    paired = paired.set_index(pd.DatetimeIndex(paired["tgt_start"]))
    paired = research_only(paired)

    n = len(paired)
    if n < 100:
        return {"hid": h.hid, "group": h.group, "n": n, "status": "too_few_obs"}

    ols = _ols_one_sided(paired["pred_ret"], paired["tgt_ret"], h.sign)

    # Cross-check with the project's primary economic test: sign(pred)*tgt,
    # oriented to the pre-registered direction, blocked by day. horizon_bars=1
    # because there is exactly one non-overlapping observation per day.
    oriented = (h.sign * paired["pred_ret"]).rename("oriented")
    src_res = signed_return_test(
        oriented, paired["tgt_ret"],
        block_by=paired.index.date, horizon_bars=1,
    )

    return {
        "hid": h.hid,
        "group": h.group,
        "predictor": h.predictor,
        "target": h.target,
        "sign": h.sign,
        "tradeable": h.tradeable,
        "n": n,
        "median_gap_h": float(paired["gap_hours"].median()),
        "slope": ols["slope"],
        "corr": ols["r"],
        "t_ols": ols["t_stat"],
        "p_one_sided": ols["p_one_sided"],
        "p_two_sided": ols["p_two_sided"],
        "sign_matches": ols["sign_matches"],
        "t_signed": src_res["t_stat"],
        "mechanism": h.mechanism,
        "status": "ok",
    }


def benjamini_hochberg(pvals: pd.Series, q: float = 0.10) -> pd.Series:
    """
    BH step-up: returns a boolean Series (True = reject null at FDR q).

    Sort p ascending; the largest rank k with p_(k) <= (k/m) q sets the
    threshold; reject all hypotheses with p <= p_(k). Controls the expected
    proportion of false discoveries among rejections at q.
    """
    p = pvals.dropna().sort_values()
    m = len(p)
    if m == 0:
        return pd.Series(dtype=bool)
    ranks = np.arange(1, m + 1)
    thresh = ranks / m * q
    passed = p.values <= thresh
    kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    cutoff = p.values[kmax - 1] if kmax > 0 else -np.inf
    return (pvals <= cutoff).reindex(pvals.index)


def run_battery(q: float = 0.10) -> pd.DataFrame:
    """Run the whole pre-registered battery on research data and apply BH-FDR."""
    rows = [run_hypothesis(h) for h in BATTERY]
    df = pd.DataFrame(rows)

    ok = df["status"] == "ok"
    # FDR is applied over the TRADEABLE cells only — the contemporaneous
    # controls are diagnostics, not hypotheses under test.
    testable = ok & df["tradeable"]
    df["bh_reject"] = False
    df.loc[testable, "bh_reject"] = benjamini_hochberg(
        df.loc[testable, "p_one_sided"], q=q
    )

    n_tradeable = int(testable.sum())
    df.attrs["q"] = q
    df.attrs["n_tradeable"] = n_tradeable
    df.attrs["bonferroni_alpha_cumulative"] = 0.05 / (PRIOR_CONFIGURATIONS + n_tradeable)
    return df.sort_values("p_one_sided", na_position="last").reset_index(drop=True)


if __name__ == "__main__":
    from src.logging_setup import setup_logging

    setup_logging()
    res = run_battery()
    cols = ["hid", "group", "predictor", "target", "sign", "n", "median_gap_h",
            "corr", "t_ols", "p_one_sided", "t_signed", "sign_matches", "bh_reject"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(res[cols].to_string(index=False))
    print(f"\nBH-FDR q={res.attrs['q']} over {res.attrs['n_tradeable']} tradeable cells")
    print(f"Bonferroni-for-context alpha = {res.attrs['bonferroni_alpha_cumulative']:.2e}")
    print(f"survivors: {res.loc[res['bh_reject'], 'hid'].tolist()}")

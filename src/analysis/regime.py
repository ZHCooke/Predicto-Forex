"""
regime.py

Conditional testing: does a feature work in SOME market states even though it
looks like nothing on average?

THE GAP THIS CLOSES. Every test so far has been unconditional. If momentum
works in trending regimes and reversion works in ranging ones, an unconditional
test averages them to roughly zero — which is exactly the pattern session 11
observed across 82 features. A signal that flips sign by regime is
indistinguishable from noise to every test we have run.

This is a genuine methodological gap rather than a data problem, which is why
it is worth closing before buying anything.

THE DANGER, STATED UP FRONT. Splitting by regime multiplies the number of tests
and is a very efficient way to manufacture false positives: with enough
subgroups something always looks significant. Three defences are built in:

1. Regimes are defined by EXOGENOUS, pre-specified variables (volatility, the
   yield curve, VIX) — never by the outcome, and never chosen by scanning for
   what works.
2. Every regime split is counted in the multiple-comparison correction.
3. `interaction_test` reports whether the DIFFERENCE between regimes is itself
   significant, not merely whether one subgroup happens to look good. A feature
   that "works in low vol" is only interesting if low vol is reliably different
   from high vol.

Regime assignment is causal: a bar's regime is determined by information
available strictly before it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.scoring import signed_return_test

log = logging.getLogger(__name__)


def quantile_regime(
    series: pd.Series, n_regimes: int = 2, window: int = 250, labels: list[str] | None = None
) -> pd.Series:
    """
    Assign each bar to a regime by where the conditioning variable sits in its
    own TRAILING distribution.

    Trailing, not full-sample: using the whole sample's quantiles would let a
    bar's regime depend on the future, which is lookahead of the subtlest kind
    — the model would "know" that 2020 was a high-vol year before it happened.

    IMPORTANT SEMANTIC: this measures "high RELATIVE TO RECENT HISTORY", not
    "high in absolute terms". A sustained shift to a new volatility level shows
    up as high only while the trailing window still remembers the old level;
    once the window has rolled over, the new level becomes the new normal and
    the regime reverts to neutral. That is deliberate — it adapts to structural
    change, and an absolute threshold would have to be calibrated on the full
    sample, reintroducing lookahead. But it means this answers "is vol
    unusually high for this market right now", not "is vol high".
    """
    if n_regimes < 2:
        raise ValueError("need at least two regimes")

    roll = series.rolling(window, min_periods=window // 2)
    # Rank of the current value within its trailing window, in [0, 1].
    pct = roll.apply(lambda w: (w.iloc[-1] > w[:-1]).mean() if len(w) > 1 else np.nan, raw=False)
    # Shift so the regime for bar t uses only information through t-1.
    pct = pct.shift(1)

    edges = np.linspace(0, 1, n_regimes + 1)[1:-1]
    reg = pd.Series(np.digitize(pct.to_numpy(), edges), index=series.index, dtype=float)
    reg[pct.isna()] = np.nan

    if labels:
        if len(labels) != n_regimes:
            raise ValueError("labels must match n_regimes")
        return reg.map({i: labels[i] for i in range(n_regimes)})
    return reg


def realized_vol_regime(returns: pd.Series, window: int = 20, **kwargs) -> pd.Series:
    """Regime by the asset's own trailing realized volatility."""
    vol = returns.rolling(window).std()
    return quantile_regime(vol, **kwargs)


def conditional_test(
    feature: pd.Series,
    forward_returns: pd.Series,
    regime: pd.Series,
    block_by: pd.Series | None = None,
) -> pd.DataFrame:
    """Run the signed-return test separately within each regime."""
    df = pd.DataFrame(
        {"f": feature, "r": forward_returns, "g": regime}
    ).dropna()
    if block_by is not None:
        df["b"] = pd.Series(block_by).reindex(df.index)

    rows = []
    for g, sub in df.groupby("g"):
        if len(sub) < 200:
            continue
        try:
            res = signed_return_test(
                sub["f"], sub["r"], block_by=sub["b"] if block_by is not None else None
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("regime %s failed: %s", g, exc)
            continue
        rows.append({"regime": g, "n": len(sub), **res})
    return pd.DataFrame(rows)


def interaction_test(
    feature: pd.Series,
    forward_returns: pd.Series,
    regime: pd.Series,
    block_by: pd.Series | None = None,
) -> dict:
    """
    Test whether a feature behaves DIFFERENTLY across two regimes.

    This is the honest question. "Feature X works in low volatility" is a claim
    about a subgroup, and subgroups always contain one that looks good. The
    defensible claim is that the two regimes genuinely differ — which is what a
    two-sample test on the per-observation signed returns asks.

    Only defined for exactly two regimes; more than that is a fishing licence.
    """
    df = pd.DataFrame({"f": feature, "r": forward_returns, "g": regime}).dropna()
    groups = sorted(df["g"].unique())
    if len(groups) != 2:
        raise ValueError(f"interaction_test needs exactly 2 regimes, got {len(groups)}")

    df["pnl"] = np.sign(df["f"]) * df["r"]
    if block_by is not None:
        b = pd.Series(block_by).reindex(df.index)
        df = df.groupby([b, "g"], as_index=False)["pnl"].mean()

    a = df.loc[df["g"] == groups[0], "pnl"]
    c = df.loc[df["g"] == groups[1], "pnl"]
    if len(a) < 2 or len(c) < 2:
        raise ValueError("not enough observations in one regime")

    t, p = stats.ttest_ind(a, c, equal_var=False)
    return {
        "regime_a": groups[0],
        "regime_b": groups[1],
        "mean_a": float(a.mean()),
        "mean_b": float(c.mean()),
        "n_a": len(a),
        "n_b": len(c),
        "t_stat": float(t),
        "p_value": float(p),
        "difference": float(a.mean() - c.mean()),
    }

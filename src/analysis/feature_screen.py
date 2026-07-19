"""
feature_screen.py

Screens every feature we have built against the signed-return test, pooled
across pairs and blocked by date, on research data only.

WHY. Until session 10 every feature was judged by the Sharpe of a strategy
built on it — a statistic so weak at this sample size that it needs ~17 years
to resolve. That means a feature with genuine directional information could
have been discarded because the STRATEGY lost to costs, which is a completely
different (and more tractable) failure. This re-runs the whole corpus with the
instrument that actually has power.

DESIGN.
- Each feature is tested per pair, then pooled, giving ~7x the observations for
  anything computable on every pair.
- Blocked by DATE, so seven correlated same-day forecasts count as one unit.
- Two-sided p-values: we do not assume the sign in advance. A significant
  negative t means the feature works inverted (e.g. reversion), which is a real
  finding, not a failure.
- Bonferroni correction applied over the full number of tests run here PLUS the
  ~84 configurations already searched in earlier sessions. Ignoring the earlier
  search would understate the correction.

WHAT THIS CANNOT TELL US. Whether anything survives costs. A feature can carry
real directional information and still lose money after spread — that is the
second filter, and it stays separate on purpose.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.holdout import research_only
from src.backtest.scoring import signed_return_test
from src.features.build_features import build_features, log_returns, resample_bars
from src.features.currency_strength import (
    PAIR_LEGS,
    build_strength_features,
    load_pair_returns,
)
from src.ingest.validate_raw import load_raw

log = logging.getLogger(__name__)

MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]

# Configurations searched in sessions 1-9, before this screen. Carried into the
# correction so the bar reflects everything we have looked at, not just today.
PRIOR_CONFIGURATIONS = 84


def _pair_frame(symbol: str, timeframe: str, rule: str) -> pd.DataFrame:
    bars = research_only(load_raw(symbol, "1h"))
    if rule:
        bars = resample_bars(bars, rule)
    return bars


def collect_price_features(rule: str, timeframe: str) -> pd.DataFrame:
    """Long-format (date, pair, feature, value, fwd_return) for price features."""
    frames = []
    for sym in MAJORS:
        bars = _pair_frame(sym, timeframe, rule)
        X = build_features(bars, timeframe=timeframe)
        fwd = log_returns(bars["close"]).shift(-1)

        d = X.copy()
        d["_fwd"] = fwd
        d = d.dropna()
        long = d.drop(columns="_fwd").stack().rename("value").reset_index()
        long.columns = ["date", "feature", "value"]
        long["fwd"] = long["date"].map(d["_fwd"])
        long["pair"] = sym
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def collect_strength_features(rule: str, timeframe: str) -> pd.DataFrame:
    """Cross-pair currency-strength features, computed for every target pair."""
    r = research_only(load_pair_returns(MAJORS, "1h", resample_rule=rule))
    fwd_all = r.shift(-1)

    frames = []
    for sym in MAJORS:
        X = build_strength_features(r, target_pair=sym)
        d = X.copy()
        d["_fwd"] = fwd_all[sym]
        d = d.dropna()
        long = d.drop(columns="_fwd").stack().rename("value").reset_index()
        long.columns = ["date", "feature", "value"]
        long["fwd"] = long["date"].map(d["_fwd"])
        long["pair"] = sym
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def collect_carry_features(rule: str, timeframe: str) -> pd.DataFrame:
    """
    Rates/carry features. EURUSD only — FRED gives us daily policy rates for
    USD and EUR, but the euro-area and JP series needed for the other pairs are
    monthly and lag ~6 months (see session 4), so they are unusable here.
    """
    from src.ingest.fetch_fred import build_rate_features

    bars = _pair_frame("EURUSD", timeframe, rule)
    X = build_rate_features(bars.index, symbol="EURUSD")
    fwd = log_returns(bars["close"]).shift(-1)

    d = X.copy()
    d["_fwd"] = fwd
    d = d.dropna()
    long = d.drop(columns="_fwd").stack().rename("value").reset_index()
    long.columns = ["date", "feature", "value"]
    long["fwd"] = long["date"].map(d["_fwd"])
    long["pair"] = "EURUSD"
    return long


def screen(long_df: pd.DataFrame, family: str, timeframe: str) -> pd.DataFrame:
    """Run the signed-return test for every feature in a long-format frame."""
    rows = []
    for feat, g in long_df.groupby("feature"):
        g = g.dropna(subset=["value", "fwd"])
        if len(g) < 200 or g["value"].std() == 0:
            continue
        try:
            res = signed_return_test(g["value"], g["fwd"], block_by=g["date"])
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s", feat, exc)
            continue
        rows.append(
            {
                "family": family,
                "timeframe": timeframe,
                "feature": feat,
                "n_raw": len(g),
                "n_eff": res["n_effective"],
                "t_stat": res["t_stat"],
                "p_value": res["p_value"],
                "mean_signed": res["mean_signed_return"],
            }
        )
    return pd.DataFrame(rows)


def run_full_screen(timeframes: tuple[tuple[str, str], ...] = (("1D", "1d"), ("4h", "4h"))) -> pd.DataFrame:
    """Screen every family at every timeframe and apply the correction."""
    out = []
    for rule, tf in timeframes:
        log.info("screening %s ...", tf)
        out.append(screen(collect_price_features(rule, tf), "price", tf))
        out.append(screen(collect_strength_features(rule, tf), "strength", tf))
        try:
            out.append(screen(collect_carry_features(rule, tf), "carry", tf))
        except Exception as exc:  # noqa: BLE001
            log.warning("carry screen failed at %s: %s", tf, exc)

    res = pd.concat([o for o in out if not o.empty], ignore_index=True)

    n_tests = len(res) + PRIOR_CONFIGURATIONS
    res["bonferroni_alpha"] = 0.05 / n_tests
    res["significant"] = res["p_value"] < res["bonferroni_alpha"]
    res["abs_t"] = res["t_stat"].abs()
    return res.sort_values("abs_t", ascending=False).reset_index(drop=True)

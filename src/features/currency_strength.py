"""
currency_strength.py

Decomposes a basket of FX pairs into per-CURRENCY strength, then builds
features from it.

THE IDEA (CLAUDE.md s7, s9.1). EURUSD is a ratio, so a move tells you the euro
strengthened relative to the dollar — but not which one moved. Watch seven
pairs at once and you can separate them: if every USD pair moves together it is
a dollar move, not a euro move. This is the ONLY genuinely exogenous
information we have added to the price side; every other price feature is a
different transform of the same single series.

METHOD. Adapted from the Elo idea in PL-Tennis-Module (each pair's move is a
"match" between two currencies) but solved algebraically rather than by
sequential Elo update, because unlike tennis we observe the exact margin, not
just who won:

    for each pair p with base b and quote q:   r_p = s_b - s_q

That is a linear system in the per-currency strengths s. With 7 pairs and 8
currencies it is rank-deficient by exactly one — only differences are
identified, since adding a constant to every currency changes nothing. We pin
it with sum(s) = 0, which makes `s` a set of returns RELATIVE TO THE BASKET
and gives the system a unique solution.

The Elo contribution survives as the rating layer: `strength_ratings()`
accumulates these per-bar strengths with exponential decay, which is what an
Elo K-factor does — recent matches matter more than old ones.

CAUSALITY. Strength at bar t is derived from the return INTO bar t
(close[t]/close[t-1]), which is known at t. So s(t) is a legitimate feature for
predicting t -> t+1. Nothing here looks forward; `tests/test_currency_strength.py`
verifies that by corrupting the future.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Quoting convention: pair -> (base, quote). A positive pair return means the
# base strengthened against the quote.
PAIR_LEGS: dict[str, tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCHF": ("USD", "CHF"),
    "USDCAD": ("USD", "CAD"),
}

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CHF", "CAD"]


def _design_matrix(
    pairs: list[str], currencies: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the (n_pairs + 1) x n_currencies system and its pseudo-inverse.

    The extra row is the sum(s) = 0 identification constraint. Without it the
    system is singular: only relative strength is observable, so some anchor
    must be chosen. Anchoring on the basket mean (rather than on USD) keeps the
    dollar itself free to move, which is the entire point — a "dollar factor"
    is only meaningful if USD is allowed a strength of its own.
    """
    idx = {c: i for i, c in enumerate(currencies)}
    A = np.zeros((len(pairs) + 1, len(currencies)))
    for r, p in enumerate(pairs):
        base, quote = PAIR_LEGS[p]
        A[r, idx[base]] = 1.0
        A[r, idx[quote]] = -1.0
    A[-1, :] = 1.0  # sum(s) = 0
    return A, np.linalg.pinv(A)


def currency_returns(
    pair_returns: pd.DataFrame, currencies: list[str] | None = None
) -> pd.DataFrame:
    """
    Per-currency return per bar, from a frame of pair returns.

    `pair_returns` columns must be pair names present in PAIR_LEGS. Returns a
    frame indexed identically with one column per currency, each row summing to
    zero (returns relative to the basket).

    Least squares handles both an over- and under-determined basket, so this
    works with any subset of pairs — though a currency touched by only one pair
    is weakly identified and one touched by none is undefined.
    """
    pairs = [p for p in pair_returns.columns if p in PAIR_LEGS]
    if not pairs:
        raise ValueError(f"no recognised pairs in {list(pair_returns.columns)}")

    if currencies is None:
        touched = {c for p in pairs for c in PAIR_LEGS[p]}
        currencies = [c for c in CURRENCIES if c in touched]

    unreachable = set(currencies) - {c for p in pairs for c in PAIR_LEGS[p]}
    if unreachable:
        raise ValueError(f"currencies not covered by any supplied pair: {sorted(unreachable)}")

    A, A_pinv = _design_matrix(pairs, currencies)

    R = pair_returns[pairs].to_numpy(dtype=float)
    # Append the zero right-hand side for the sum-to-zero constraint.
    R_aug = np.hstack([R, np.zeros((len(R), 1))])
    S = R_aug @ A_pinv.T

    out = pd.DataFrame(S, index=pair_returns.index, columns=currencies)
    # Rows where any pair was missing are not trustworthy — the decomposition
    # would silently attribute the gap to the remaining currencies.
    out[pair_returns[pairs].isna().any(axis=1)] = np.nan
    return out


def strength_ratings(
    ccy_returns: pd.DataFrame, halflife: float = 20.0
) -> pd.DataFrame:
    """
    Exponentially-decayed cumulative strength — the Elo rating layer.

    An Elo K-factor makes recent matches count more than old ones; an EWMA of
    per-bar strengths does the same thing continuously. `halflife` is in bars:
    at 20 on daily data, form from a month ago carries half the weight of today.

    Causal by construction: pandas' ewm uses only past and current observations.
    """
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    return ccy_returns.ewm(halflife=halflife, ignore_na=False).mean()


def build_strength_features(
    pair_returns: pd.DataFrame,
    target_pair: str = "EURUSD",
    halflives: tuple[float, ...] = (5.0, 20.0, 60.0),
) -> pd.DataFrame:
    """
    Cross-sectional currency features for predicting `target_pair`.

    All columns are prefixed `f_` so they drop into the existing feature matrix.
    """
    if target_pair not in PAIR_LEGS:
        raise KeyError(f"{target_pair!r} not in PAIR_LEGS")

    base, quote = PAIR_LEGS[target_pair]
    ccy = currency_returns(pair_returns)

    if base not in ccy.columns or quote not in ccy.columns:
        raise ValueError(f"{target_pair} legs {base}/{quote} not covered by the basket")

    out = pd.DataFrame(index=pair_returns.index)

    for hl in halflives:
        r = strength_ratings(ccy, halflife=hl)
        h = int(hl)
        # The headline feature: how strong the base is versus the quote, judged
        # against the whole basket rather than against each other alone.
        out[f"f_ccy_diff_{h}"] = r[base] - r[quote]
        out[f"f_ccy_base_{h}"] = r[base]
        out[f"f_ccy_quote_{h}"] = r[quote]
        # The "dollar factor" of the literature: USD strength versus the basket.
        if "USD" in r.columns:
            out[f"f_dollar_{h}"] = r["USD"]
        # Cross-sectional dispersion of the RATINGS: how concentrated the
        # current regime is. High when one currency is persistently dominating
        # the basket (a sustained dollar trend pushes USD far from the others);
        # low when moves are noisy and wash out of the EWMA. So this reads as
        # trend strength across the whole basket, NOT as idiosyncrasy — a
        # distinction single-pair features cannot see at all.
        out[f"f_ccy_dispersion_{h}"] = r.std(axis=1)

    # Rank of each leg within the basket, scaled to [0, 1]. Rank is robust to
    # the scale of returns and is the form the cross-sectional FX factor
    # literature usually trades.
    ranks = ccy.rank(axis=1, pct=True)
    out["f_ccy_rank_base"] = ranks[base]
    out["f_ccy_rank_quote"] = ranks[quote]

    # Rolling z-score of the strength differential.
    #
    # Motivated by a specific diagnostic, not by fishing: the raw f_ccy_diff_5
    # averaged an IC of about -0.047 WITHIN each year but only -0.016 pooled
    # across years. That gap is the signature of level drift — each year sits
    # at a different baseline, so a pooled correlation partly measures the
    # drift rather than the signal. Normalising against a trailing window
    # removes the baseline and should recover the within-year relationship.
    #
    # Strictly trailing (no centering, no future data), so it stays causal.
    for hl in halflives:
        h = int(hl)
        window = max(20, int(hl) * 4)
        d = out[f"f_ccy_diff_{h}"]
        roll = d.rolling(window, min_periods=window // 2)
        out[f"f_ccy_diff_z_{h}"] = (d - roll.mean()) / roll.std()

    assert all(c.startswith("f_") for c in out.columns)
    return out


def load_pair_returns(
    symbols: list[str], timeframe: str = "1h", resample_rule: str | None = None
) -> pd.DataFrame:
    """
    Load several pairs and return their aligned log returns.

    Aligned on the INTERSECTION of timestamps: a bar where any pair is missing
    cannot be decomposed, and dropping it is safer than imputing a return.
    """
    from src.features.build_features import log_returns, resample_bars
    from src.ingest.validate_raw import load_raw

    closes = {}
    for sym in symbols:
        df = load_raw(sym, timeframe)
        if resample_rule:
            df = resample_bars(df, resample_rule)
        closes[sym] = df["close"]

    px = pd.DataFrame(closes).dropna()
    log.info("aligned %d pairs on %d common bars", len(symbols), len(px))
    return pd.DataFrame({s: log_returns(px[s]) for s in symbols}).dropna()

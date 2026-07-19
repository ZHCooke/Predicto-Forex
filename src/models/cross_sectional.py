"""
cross_sectional.py

Trades the currency-strength reversion signal across ALL pairs at once as a
single portfolio, rather than as one strategy per pair.

WHY THIS IS THE RIGHT MOVE (CLAUDE.md session 8). The binding constraint is no
longer costs or model class — it is sample size. Daily EURUSD gives ~2,400
observations, and no amount of calibration or ensembling manufactures more
information from that. Running the SAME hypothesis across seven pairs gives
roughly 7x the observations while remaining ONE hypothesis, so it does not
inflate the multiple-comparisons problem the way seven separate backtests would.

It is also the form the FX factor literature actually uses: carry, momentum and
value are documented as CROSS-SECTIONAL effects (rank currencies, go long the
strong and short the weak) and are consistently weaker as time-series signals
on a single pair, which is exactly the weaker formulation we have been running
until now.

TWO STRUCTURAL ADVANTAGES BEYOND SAMPLE SIZE.

1. Net neutrality. Every pair we hold is USD-based or USD-quoted, so
   independent per-pair bets accumulate a large accidental aggregate position.
   Demeaning the scores makes the currency exposures sum to zero: for every
   long there is an offsetting short.

   TERMINOLOGY WARNING — "dollar-neutral" means two different things and
   conflating them is easy. In the equity long/short sense it means the net
   NOTIONAL cancels, which is what demeaning achieves and what this module
   implements. It does NOT mean zero exposure to USD specifically: USD is
   scored like any other currency, so if the dollar has run up, reversion will
   short it. That is a deliberate view, not a leak. `portfolio_diagnostics`
   reports both quantities separately for exactly this reason.

2. Netting redundant legs. Positions are expressed per CURRENCY and then
   converted to the minimum set of pair trades. Holding EURUSD long and GBPUSD
   short is partly a EUR/GBP view with the dollar legs cancelling; trading both
   legs separately pays two spreads for one economic position. This directly
   addresses the cost warning in s7 — without netting, N pairs cost ~N spreads.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.currency_strength import PAIR_LEGS, currency_returns, strength_ratings

log = logging.getLogger(__name__)


def cross_sectional_scores(
    ccy_ratings: pd.DataFrame, demean: bool = True
) -> pd.DataFrame:
    """
    Convert per-currency strength ratings into cross-sectional scores.

    Demeaning each row is what makes the book dollar-neutral in aggregate: the
    scores sum to zero, so for every currency we are long there is an equal and
    opposite short. Without it a broad risk-on move would put us the same
    direction in everything at once.
    """
    scores = ccy_ratings.copy()
    if demean:
        scores = scores.sub(scores.mean(axis=1), axis=0)
    return scores


def currency_positions_from_scores(
    scores: pd.DataFrame, reversion: bool = True, gross_exposure: float = 1.0
) -> pd.DataFrame:
    """
    Per-currency target weights from cross-sectional scores.

    `reversion=True` FADES recent strength (short what has run up), which is the
    direction the research-period evidence supports: `f_ccy_diff_5` carries a
    per-year mean IC of about -0.055 with 88% sign agreement.

    Weights are normalised so gross exposure (sum of absolute weights) is
    constant, keeping risk comparable across dates regardless of how dispersed
    the scores happen to be on any given day.
    """
    w = -scores if reversion else scores.copy()

    gross = w.abs().sum(axis=1)
    w = w.div(gross.replace(0, np.nan), axis=0) * gross_exposure
    return w.fillna(0.0)


def currency_to_pair_positions(
    ccy_weights: pd.DataFrame, pairs: list[str]
) -> pd.DataFrame:
    """
    Convert per-currency weights into per-pair positions, netting redundancy.

    A position in pair p with legs (base, quote) gives +1 unit of base exposure
    and -1 of quote. We solve the least-squares system

        A @ pair_positions ~= currency_weights

    where A is the currency-by-pair incidence matrix. Least squares finds the
    MINIMUM-NORM set of pair trades reproducing the desired currency exposure,
    which is precisely the netting we want: redundant legs cancel instead of
    being traded twice.

    The reproduction is exact only if the target weights lie in the span of the
    available pairs. With 7 pairs and 8 currencies the span is 7-dimensional and
    demeaned weights sum to zero, so the fit is exact for dollar-neutral books —
    `portfolio_diagnostics` checks the residual rather than assuming it.
    """
    currencies = list(ccy_weights.columns)
    idx = {c: i for i, c in enumerate(currencies)}

    A = np.zeros((len(currencies), len(pairs)))
    for j, p in enumerate(pairs):
        base, quote = PAIR_LEGS[p]
        if base in idx:
            A[idx[base], j] = 1.0
        if quote in idx:
            A[idx[quote], j] = -1.0

    A_pinv = np.linalg.pinv(A)
    pos = ccy_weights.to_numpy() @ A_pinv.T
    return pd.DataFrame(pos, index=ccy_weights.index, columns=pairs)


def build_cross_sectional_positions(
    pair_returns: pd.DataFrame,
    halflife: float = 5.0,
    reversion: bool = True,
    gross_exposure: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: pair returns -> currency strength -> scores -> pair positions.

    Returns (pair_positions, currency_weights).

    CAUSALITY: ratings at bar t use returns through t only, and the engine
    applies its own one-bar shift before computing P&L, so a position derived
    from bar t earns the move from t to t+1.
    """
    pairs = [p for p in pair_returns.columns if p in PAIR_LEGS]
    ccy = currency_returns(pair_returns[pairs])
    ratings = strength_ratings(ccy, halflife=halflife)

    scores = cross_sectional_scores(ratings)
    ccy_w = currency_positions_from_scores(scores, reversion, gross_exposure)
    pair_pos = currency_to_pair_positions(ccy_w, pairs)

    return pair_pos, ccy_w


def portfolio_returns(
    pair_positions: pd.DataFrame, pair_returns: pd.DataFrame
) -> pd.Series:
    """
    Portfolio P&L, applying the one-bar execution shift.

    Same convention as the single-pair engine: a position formed from bar t's
    information is held over t -> t+1, so positions are shifted before being
    multiplied by returns.
    """
    common = pair_positions.columns.intersection(pair_returns.columns)
    pos = pair_positions[common].shift(1)
    return (pos * pair_returns[common]).sum(axis=1).dropna()


def portfolio_turnover(pair_positions: pd.DataFrame) -> pd.Series:
    """Total absolute position change per bar, summed across pairs."""
    d = pair_positions.diff().abs().sum(axis=1)
    d.iloc[0] = pair_positions.iloc[0].abs().sum()
    return d


def portfolio_diagnostics(
    pair_positions: pd.DataFrame, ccy_weights: pd.DataFrame
) -> dict:
    """
    Sanity checks on the book. Reported alongside performance because a
    cross-sectional strategy can look fine on P&L while carrying an unintended
    directional bet.
    """
    pairs = list(pair_positions.columns)
    currencies = list(ccy_weights.columns)
    idx = {c: i for i, c in enumerate(currencies)}

    A = np.zeros((len(currencies), len(pairs)))
    for j, p in enumerate(pairs):
        base, quote = PAIR_LEGS[p]
        if base in idx:
            A[idx[base], j] = 1.0
        if quote in idx:
            A[idx[quote], j] = -1.0

    achieved = pair_positions.to_numpy() @ A.T
    residual = np.abs(achieved - ccy_weights.to_numpy()).max()

    # NET exposure across all currencies: must be ~0 (every long offset by a
    # short). This is the "dollar-neutral" property in the equity sense.
    net_total = pd.Series(achieved.sum(axis=1), index=pair_positions.index)

    # Exposure to USD specifically. This is NOT expected to be zero — it is a
    # deliberate directional view on the dollar, and the two must not be
    # confused. Reported so an unintended dollar tilt is at least visible.
    usd_exposure = (
        pd.Series(achieved[:, idx["USD"]], index=pair_positions.index)
        if "USD" in idx else pd.Series(dtype=float)
    )

    return {
        "max_replication_residual": float(residual),
        "mean_gross_pair_exposure": float(pair_positions.abs().sum(axis=1).mean()),
        "mean_gross_ccy_exposure": float(ccy_weights.abs().sum(axis=1).mean()),
        "max_abs_net_exposure": float(net_total.abs().max()),
        "mean_abs_usd_exposure": float(usd_exposure.abs().mean()) if len(usd_exposure) else np.nan,
        "netting_ratio": float(
            pair_positions.abs().sum(axis=1).mean()
            / max(ccy_weights.abs().sum(axis=1).mean(), 1e-12)
        ),
    }

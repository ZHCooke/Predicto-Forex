"""
scoring.py

Proper scoring rules (log loss, Brier) for directional forecasts, plus tests of
whether a model beats a no-skill baseline.

WHY THIS EXISTS — IT SOLVES A POWER PROBLEM SHARPE CANNOT.

Sharpe compresses an entire year of trading into one number, so eight years of
data gives roughly eight effective observations. The arithmetic is brutal: to
show a Sharpe of 0.5 is distinguishable from luck at 95% you need about
17 years. We have 11.5. That is why every result in this project has had a
confidence interval straddling zero, and no amount of modelling fixes it.

Log loss is scored PER PREDICTION. Every forecast is graded individually, so
the sample is the number of forecasts, not the number of years.

And the cross-sectional structure now works FOR us. For Sharpe, seven pairs
collapse into one portfolio return per day — no extra observations (a point I
originally got wrong and corrected in session 9). For scoring, each pair's
forecast is graded separately: 7 pairs x ~2,400 days = ~16,800 graded
predictions. That is a genuine order-of-magnitude gain in power.

WHAT IT CAN AND CANNOT SETTLE.

CAN: "does this model contain real predictive information?" — a question about
     forecast quality, answerable with high confidence on this data.
CANNOT: "does it make money?" — that depends on costs, sizing and the size of
     the moves it gets right, none of which log loss sees. A model can predict
     direction well on tiny moves and still lose after spread.

So the two are complementary, not substitutes: log loss establishes that signal
EXISTS; the backtest establishes whether it SURVIVES costs. A strategy needs
both, and until now we have only been able to test the second — with a tool too
blunt to give an answer.

DIAGNOSTIC, NOT OBJECTIVE — AND THIS IS DELIBERATE.

PL-AFL-Module computes log loss, Brier and ROC-AUC per round
(`backtesting/metrics.py::summarise_round`) and accumulates them, but it never
FITS to them — its models are regressions on margin, with probabilities coming
from calibration afterwards. That choice is right for us too, and for a sharper
reason: log loss weights every observation equally, whereas P&L weights by the
SIZE of the move. A model tuned to maximise log loss would happily buy accuracy
on many tiny moves at the cost of the few large ones that actually pay for the
spread. Optimising for it would therefore be optimising against profitability.

Use it to answer "is there signal here at all", which our returns-based
confidence intervals are too weak to settle. Keep fitting to returns.

The per-period + cumulative structure below mirrors AFL's round-by-round
reporting, because the TRAJECTORY is the diagnostic: a real edge accumulates
steadily, an artifact shows one lucky stretch and nothing either side.

DEPENDENCE CAVEAT. The seven pairs share currency factors, so their forecasts
are correlated and the effective sample is smaller than the raw count. Naive
per-observation standard errors would overstate significance. `paired_test`
therefore blocks by DATE, treating each day as one unit, which is conservative
and correct.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

EPS = 1e-15
COIN_FLIP_LOG_LOSS = float(np.log(2))  # 0.6931 — the no-skill benchmark


def returns_to_probability(
    predicted_return: pd.Series, predicted_vol: pd.Series | float
) -> pd.Series:
    """
    Convert a predicted return into P(up), via the normal CDF of the
    predicted return scaled by volatility.

    This mirrors the margin -> probability step in PL-AFL-Module's calibrator
    (sigmoid of margin/sigma); the normal CDF is the same idea with a
    distributional justification, since we are modelling returns rather than
    sporting margins.
    """
    vol = (
        pd.Series(predicted_vol, index=predicted_return.index)
        if np.isscalar(predicted_vol)
        else predicted_vol.reindex(predicted_return.index)
    )
    z = predicted_return / vol.replace(0, np.nan)
    p = pd.Series(stats.norm.cdf(z.fillna(0.0)), index=predicted_return.index)
    return p.clip(0.001, 0.999)


def log_loss(probabilities: pd.Series, outcomes: pd.Series) -> pd.Series:
    """
    Per-observation log loss. `outcomes` is 1 if the move was up, else 0.

    Lower is better. ln(2) = 0.6931 is the score of always saying 50/50, so
    anything below that indicates skill and anything above indicates the model
    is worse than useless — confidently wrong rather than merely uninformed.
    """
    p = probabilities.clip(EPS, 1 - EPS)
    y = outcomes.reindex(p.index)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_score(probabilities: pd.Series, outcomes: pd.Series) -> pd.Series:
    """Per-observation squared error. Less sensitive than log loss to
    confident mistakes, which makes it a useful cross-check."""
    p = probabilities.clip(0, 1)
    return (p - outcomes.reindex(p.index)) ** 2


def running_log_loss(
    probabilities: pd.Series, outcomes: pd.Series, window: int | None = None
) -> pd.DataFrame:
    """
    Cumulative and rolling log loss over time, against the coin-flip baseline.

    The `edge` column is (baseline - model): positive means skill. Watching it
    over time is the point — a real signal accumulates edge steadily, whereas
    an artifact shows one lucky stretch and flat or negative elsewhere. That is
    exactly the distinction our confidence intervals have been too weak to draw.
    """
    ll = log_loss(probabilities, outcomes).dropna()
    out = pd.DataFrame({"log_loss": ll})
    out["cumulative"] = ll.expanding().mean()
    out["edge"] = COIN_FLIP_LOG_LOSS - out["cumulative"]
    if window:
        out[f"rolling_{window}"] = ll.rolling(window).mean()
        out[f"rolling_edge_{window}"] = COIN_FLIP_LOG_LOSS - out[f"rolling_{window}"]
    return out


def paired_test(
    model_loss: pd.Series,
    baseline_loss: pd.Series | float = COIN_FLIP_LOG_LOSS,
    block_by: pd.Index | pd.Series | None = None,
) -> dict:
    """
    Test whether the model's loss is genuinely below the baseline's.

    A PAIRED test (Diebold-Mariano in spirit): we compare losses observation by
    observation, so anything affecting both models equally cancels out. This is
    far more powerful than comparing two separately-estimated means.

    `block_by` groups observations that are not independent — pass the DATE for
    cross-sectional forecasts, so seven correlated same-day predictions count as
    one unit rather than seven. Without this the p-value would be badly
    overstated.
    """
    base = (
        pd.Series(baseline_loss, index=model_loss.index)
        if np.isscalar(baseline_loss)
        else baseline_loss.reindex(model_loss.index)
    )
    diff = (base - model_loss).dropna()  # positive = model better
    if diff.empty:
        raise ValueError("no overlapping observations to test")

    if block_by is not None:
        blocks = pd.Series(block_by, index=model_loss.index).reindex(diff.index)
        diff = diff.groupby(blocks).mean()

    n = len(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan

    # Guard against a degenerate near-constant loss series.
    #
    # `se > 0` is NOT sufficient: the standard deviation of a constant series
    # comes out around 1e-17 in floating point rather than exactly zero, which
    # slips through and produces t = 6.5e16 with p = 0.0 — a fabricated
    # "highly significant" result from data containing no information at all.
    # The threshold must be RELATIVE to the magnitude being tested.
    scale = max(abs(mean), float(np.abs(diff).mean()), EPS)
    degenerate = not np.isfinite(se) or se <= scale * 1e-10

    t_stat = np.nan if degenerate else mean / se
    p_two = (
        float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
        if np.isfinite(t_stat) else np.nan
    )

    return {
        "n_effective": n,
        "mean_edge": mean,
        "std_error": se,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": p_two,
        "ci_lo": mean - 1.96 * se if np.isfinite(se) else np.nan,
        "ci_hi": mean + 1.96 * se if np.isfinite(se) else np.nan,
        "model_better": bool(mean > 0),
    }


def directional_accuracy(probabilities: pd.Series, outcomes: pd.Series) -> float:
    """Fraction of forecasts on the correct side of 50/50."""
    p = probabilities.dropna()
    y = outcomes.reindex(p.index)
    return float(((p > 0.5).astype(int) == y).mean())


def safe_roc_auc(outcomes: pd.Series, probabilities: pd.Series) -> float:
    """
    ROC-AUC, returning NaN when a period contains only one outcome class.

    Mirrors PL-AFL-Module's `safe_roc_auc`: short periods routinely have all-up
    or all-down days, where AUC is undefined and sklearn raises.
    """
    from sklearn.metrics import roc_auc_score

    y = outcomes.dropna()
    p = probabilities.reindex(y.index)
    if y.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def summarise_period(
    probabilities: pd.Series, outcomes: pd.Series, label: object = None
) -> dict:
    """
    One-row scoring summary for a single period.

    Deliberately the same shape as PL-AFL-Module's `summarise_round`: accuracy,
    log loss, Brier and ROC-AUC together, because they fail in different ways.
    Accuracy ignores confidence, Brier is forgiving of confident errors, log
    loss punishes them hard, and AUC ignores calibration entirely. Agreement
    between them is far more convincing than any one in isolation.
    """
    p = probabilities.dropna()
    y = outcomes.reindex(p.index).dropna()
    p = p.reindex(y.index)
    if p.empty:
        return {"label": label, "n": 0}

    ll = log_loss(p, y)
    return {
        "label": label,
        "n": int(len(p)),
        "accuracy": directional_accuracy(p, y),
        "log_loss": float(ll.mean()),
        "log_loss_edge": COIN_FLIP_LOG_LOSS - float(ll.mean()),
        "brier": float(brier_score(p, y).mean()),
        "roc_auc": safe_roc_auc(y, p),
        "mean_prob": float(p.mean()),
    }


def running_scores(
    probabilities: pd.Series,
    outcomes: pd.Series,
    period: str = "YE",
) -> pd.DataFrame:
    """
    Per-period scores plus cumulative accuracy and log loss.

    The cumulative columns follow AFL's pattern (weighted by observation count,
    not a mean of means, which would over-weight short periods). Reading the
    per-period column is the point: an edge present in one year and absent in
    seven is not an edge.
    """
    p = probabilities.dropna()
    y = outcomes.reindex(p.index).dropna()
    p = p.reindex(y.index)

    rows = [
        summarise_period(p[g.index], y[g.index], label=str(name)[:10])
        for name, g in p.groupby(pd.Grouper(freq=period))
        if len(g) > 0
    ]
    out = pd.DataFrame([r for r in rows if r.get("n", 0) > 0]).set_index("label")
    if out.empty:
        return out

    # Count-weighted cumulative figures.
    out["cum_n"] = out["n"].cumsum()
    out["cum_accuracy"] = (out["accuracy"] * out["n"]).cumsum() / out["cum_n"]
    out["cum_log_loss"] = (out["log_loss"] * out["n"]).cumsum() / out["cum_n"]
    out["cum_edge"] = COIN_FLIP_LOG_LOSS - out["cum_log_loss"]
    return out


def score_report(
    probabilities: pd.Series,
    outcomes: pd.Series,
    block_by: pd.Index | pd.Series | None = None,
) -> dict:
    """Full scoring summary for one set of probabilistic forecasts."""
    ll = log_loss(probabilities, outcomes).dropna()
    test = paired_test(ll, COIN_FLIP_LOG_LOSS, block_by=block_by)
    return {
        "n_forecasts": len(ll),
        "mean_log_loss": float(ll.mean()),
        "baseline_log_loss": COIN_FLIP_LOG_LOSS,
        "log_loss_edge": COIN_FLIP_LOG_LOSS - float(ll.mean()),
        "mean_brier": float(brier_score(probabilities, outcomes).mean()),
        "accuracy": directional_accuracy(probabilities, outcomes),
        **{f"test_{k}": v for k, v in test.items()},
    }


def overlap_warning(horizon_bars: int, sampling_interval_bars: int = 1) -> str | None:
    """
    Flag the overlapping-returns trap.

    A forward return over H bars, sampled every bar, produces observations that
    share (H-1)/H of their content. Treating them as independent inflates the
    t-statistic by roughly sqrt(H).

    This is not hypothetical. COT positioning at a 20-day horizon scored
    t = -3.75 (p = 0.0002, comfortably through Bonferroni and bound for the
    holdout) when blocked by date alone. Sampled strictly non-overlapping, the
    same effect was t = -0.74, p = 0.46 — nothing. The entire result was
    manufactured by counting 2,382 overlapping observations as independent when
    there were about 120.

    Blocking by date fixes CROSS-SECTIONAL correlation (several pairs on one
    day). It does nothing for SERIAL overlap. They are different problems and
    both need handling.
    """
    if horizon_bars <= sampling_interval_bars:
        return None
    ratio = horizon_bars / sampling_interval_bars
    return (
        f"forward horizon is {horizon_bars} bars but sampled every "
        f"{sampling_interval_bars}: observations overlap ~{100 * (1 - 1 / ratio):.0f}%. "
        f"Naive standard errors are inflated by roughly sqrt({ratio:.0f}) = "
        f"{np.sqrt(ratio):.1f}x. Sample non-overlapping, or block by a period "
        f"at least {horizon_bars} bars long."
    )


def signed_return_test(
    predictions: pd.Series,
    actual_returns: pd.Series,
    block_by: pd.Index | pd.Series | None = None,
    horizon_bars: int = 1,
) -> dict:
    """
    Test whether predictions have economic value: is E[sign(pred) x return] > 0?

    THIS IS THE PRIMARY POWER TEST, and log loss is NOT. Measured directly:

        calibrated forecaster, 54% edge, n = 2,000
            accuracy test  z = 3.58
            log-loss test  z = 1.79     <- half the power

    Log loss is roughly half as sensitive as a directional test for detecting a
    calibrated edge, because a small well-calibrated edge barely moves the loss
    (edge ~ 2*(p-0.5)^2, so 54% buys only 0.003 of the 0.693 baseline) while the
    outcome noise stays large. Log loss earns its place by grading CALIBRATION —
    it punishes confident errors, which is what protects Kelly sizing — not by
    being the sharpest instrument for detecting direction.

    This test is better still than accuracy, because it weights each observation
    by the SIZE of the move. That is the economically meaningful quantity: being
    right on the big days is what pays for spread, and an accuracy test cannot
    see the difference.

    `block_by` groups correlated observations (pass the date for cross-sectional
    forecasts) exactly as in `paired_test`.
    """
    warn = overlap_warning(horizon_bars)
    if warn:
        log.warning("signed_return_test: %s", warn)

    pred = predictions.dropna()
    r = actual_returns.reindex(pred.index).dropna()
    pred = pred.reindex(r.index)

    pnl = np.sign(pred) * r
    if block_by is not None:
        blocks = pd.Series(block_by, index=predictions.index).reindex(pnl.index)
        pnl = pnl.groupby(blocks).mean()

    pnl = pnl.dropna()
    n = len(pnl)
    if n < 2:
        raise ValueError("need at least two observations")

    mean = float(pnl.mean())
    se = float(pnl.std(ddof=1) / np.sqrt(n))
    scale = max(abs(mean), float(np.abs(pnl).mean()), EPS)
    degenerate = not np.isfinite(se) or se <= scale * 1e-10

    t_stat = np.nan if degenerate else mean / se
    p_two = (
        float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)))
        if np.isfinite(t_stat) else np.nan
    )

    return {
        "n_effective": n,
        "mean_signed_return": mean,
        "std_error": se,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": p_two,
        "hit_rate": float((pnl > 0).mean()),
        "overlap_warning": warn,
    }


def required_years_for_sharpe(sharpe: float, alpha: float = 0.05) -> float:
    """
    Years of data needed for a Sharpe estimate's CI to exclude zero.

    Uses SE(S) ~= sqrt((1 + S^2/2) / T). Included because it is the number that
    reframes this whole project: at S = 0.5 it returns ~17 years, which is why
    demanding statistical significance on the RETURNS was never achievable with
    11.5 years of data — and why per-prediction scoring is the way out.
    """
    if sharpe <= 0:
        return float("inf")
    z = stats.norm.ppf(1 - alpha / 2)
    return float(z**2 * (1 + sharpe**2 / 2) / sharpe**2)

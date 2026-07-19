"""
Scoring-rule tests. The important ones are the negative controls: a random
forecaster must NOT look skilful, and correlated cross-sectional forecasts must
not be counted as independent evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.scoring import (
    COIN_FLIP_LOG_LOSS,
    brier_score,
    directional_accuracy,
    log_loss,
    paired_test,
    required_years_for_sharpe,
    returns_to_probability,
    running_log_loss,
    score_report,
    signed_return_test,
)


@pytest.fixture
def idx():
    return pd.date_range("2015-01-01", periods=2000, freq="D", tz="UTC")


def test_coin_flip_scores_exactly_ln2(idx) -> None:
    p = pd.Series(0.5, index=idx)
    y = pd.Series(np.random.default_rng(0).integers(0, 2, len(idx)), index=idx)
    assert log_loss(p, y).mean() == pytest.approx(COIN_FLIP_LOG_LOSS)


def test_perfect_forecaster_scores_near_zero(idx) -> None:
    y = pd.Series(np.random.default_rng(1).integers(0, 2, len(idx)), index=idx)
    p = y.astype(float).clip(0.001, 0.999)
    assert log_loss(p, y).mean() < 0.01


def test_confidently_wrong_is_punished_worse_than_uninformed(idx) -> None:
    """The property that makes log loss the right tool: it penalises
    confident errors far more than admitting ignorance."""
    y = pd.Series(1, index=idx)
    uninformed = log_loss(pd.Series(0.5, index=idx), y).mean()
    confident_wrong = log_loss(pd.Series(0.02, index=idx), y).mean()
    assert confident_wrong > 3 * uninformed


def test_random_forecaster_is_not_significant(idx) -> None:
    """
    NEGATIVE CONTROL. Noise must not produce a significant result — if this
    fails, every positive finding from this module is worthless.
    """
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, len(idx)), index=idx)
    p = pd.Series(rng.uniform(0.45, 0.55, len(idx)), index=idx)

    res = score_report(p, y)
    assert res["test_p_value"] > 0.05
    # A random forecaster is typically slightly WORSE than a flat 50/50.
    assert res["log_loss_edge"] < 0.01


def _weak_forecaster(n: int, edge: float, seed: int):
    """
    A realistic weak forecaster: it is right only ~(0.5 + edge) of the time, so
    per-observation loss VARIES. An always-correct forecaster has constant loss,
    zero variance and an undefined t-statistic — which is what the first version
    of this test accidentally built.
    """
    rng = np.random.default_rng(seed)
    conf = 0.5 + edge
    calls = rng.integers(0, 2, n)                       # what we predict
    correct = rng.random(n) < conf                      # whether we are right
    y = np.where(correct, calls, 1 - calls)
    p = np.where(calls == 1, conf, 1 - conf)
    return pd.Series(p), pd.Series(y)


def test_genuine_weak_skill_is_detected(idx) -> None:
    """
    The power claim: a small but real edge over ~2000 forecasts should be
    detectable, where a Sharpe test on the same data would not be.
    """
    # edge 0.08: log loss is only ~half as powerful as an accuracy test,
    # so a 54% forecaster over 2000 bars is NOT detectable this way.
    p, y = _weak_forecaster(len(idx), edge=0.08, seed=3)
    p.index, y.index = idx, idx

    res = score_report(p, y)
    assert res["log_loss_edge"] > 0
    assert res["test_p_value"] < 0.01


def test_zero_variance_loss_does_not_fake_significance() -> None:
    """
    A forecaster with perfectly constant loss gives no basis for inference.
    The test must return NaN rather than an infinite t-statistic.
    """
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    y = pd.Series(1, index=idx)
    res = paired_test(log_loss(pd.Series(0.6, index=idx), y))
    assert np.isnan(res["t_stat"]) or np.isnan(res["p_value"])


def test_blocking_reduces_effective_sample(idx) -> None:
    """
    Correlated same-day forecasts must count as one unit, not seven. Without
    blocking the p-value is overstated — the single most likely way this module
    could manufacture a false positive.
    """
    # Observations must be genuinely CORRELATED within a date, otherwise
    # blocking is a no-op: averaging k independent values then dividing by
    # sqrt(n/k) gives the same standard error. A shared per-date component is
    # what makes the naive estimate overstate significance.
    # The CORRECTNESS must be correlated within a date, not merely the
    # direction called. If the model is right on all seven pairs some days and
    # wrong on all seven on others — which is what a common-factor day looks
    # like in FX — then the LOSSES are correlated and the naive standard error
    # is genuinely too small.
    rng = np.random.default_rng(4)
    n_dates, per_date = 500, 7
    dates = np.repeat(idx[:n_dates], per_date)

    day_skill = np.repeat(rng.choice([0.15, 0.90], size=n_dates), per_date)
    calls = rng.integers(0, 2, len(dates))
    correct = rng.random(len(dates)) < day_skill
    y = pd.Series(np.where(correct, calls, 1 - calls))
    p = pd.Series(np.where(calls == 1, 0.53, 0.47))

    unblocked = paired_test(log_loss(p, y))
    blocked = paired_test(log_loss(p, y), block_by=pd.Series(dates))

    assert blocked["n_effective"] == 500
    assert unblocked["n_effective"] == 3500
    # Fewer effective observations must widen the interval — treating
    # correlated same-day forecasts as independent overstates significance.
    assert blocked["std_error"] > unblocked["std_error"]
    assert blocked["p_value"] > unblocked["p_value"]


def test_paired_test_is_symmetric_about_zero(idx) -> None:
    """A model that is WORSE than baseline must report negative edge."""
    y = pd.Series(1, index=idx)
    bad = log_loss(pd.Series(0.3, index=idx), y)
    res = paired_test(bad)
    assert res["mean_edge"] < 0
    assert not res["model_better"]


def test_running_log_loss_tracks_accumulating_edge(idx) -> None:
    y = pd.Series(1, index=idx)
    p = pd.Series(0.6, index=idx)
    run = running_log_loss(p, y, window=50)
    assert (run["edge"] > 0).all()
    assert run["cumulative"].iloc[-1] == pytest.approx(log_loss(p, y).mean())


def test_returns_to_probability_is_monotone_and_centred(idx) -> None:
    r = pd.Series(np.linspace(-0.02, 0.02, len(idx)), index=idx)
    p = returns_to_probability(r, 0.01)
    assert p.is_monotonic_increasing
    assert p.iloc[len(p) // 2] == pytest.approx(0.5, abs=0.01)
    assert p.between(0, 1).all()


def test_zero_prediction_gives_no_opinion(idx) -> None:
    p = returns_to_probability(pd.Series(0.0, index=idx), 0.01)
    assert np.allclose(p.to_numpy(), 0.5)


def test_brier_agrees_with_log_loss_on_direction(idx) -> None:
    rng = np.random.default_rng(5)
    y = pd.Series(rng.integers(0, 2, len(idx)), index=idx)
    good = pd.Series(np.where(y == 1, 0.6, 0.4), index=idx)
    bad = pd.Series(np.where(y == 1, 0.4, 0.6), index=idx)

    assert brier_score(good, y).mean() < brier_score(bad, y).mean()
    assert log_loss(good, y).mean() < log_loss(bad, y).mean()


def test_accuracy_matches_hand_count() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    p = pd.Series([0.6, 0.4, 0.7, 0.3], index=idx)
    y = pd.Series([1, 0, 0, 0], index=idx)
    assert directional_accuracy(p, y) == pytest.approx(0.75)


def test_required_years_reproduces_the_project_constraint() -> None:
    """The number that reframed this project: ~17 years to prove S = 0.5."""
    assert required_years_for_sharpe(0.5) == pytest.approx(17.3, abs=1.0)
    assert required_years_for_sharpe(1.0) < 7
    # Better strategies need less data — the relationship must be decreasing.
    assert required_years_for_sharpe(0.5) > required_years_for_sharpe(0.8)


# --- AFL-style per-period reporting ----------------------------------------

def test_summarise_period_matches_afl_shape(idx) -> None:
    """Same metric set as PL-AFL-Module's summarise_round — they fail in
    different ways, so they are reported together."""
    rng = np.random.default_rng(10)
    y = pd.Series(rng.integers(0, 2, len(idx)), index=idx)
    p = pd.Series(np.where(y == 1, 0.55, 0.45), index=idx)

    from src.backtest.scoring import summarise_period
    s = summarise_period(p, y, label="2020")
    assert set(s) >= {"n", "accuracy", "log_loss", "brier", "roc_auc", "log_loss_edge"}
    assert s["accuracy"] == pytest.approx(1.0)
    assert s["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_nan_for_single_class(idx) -> None:
    from src.backtest.scoring import safe_roc_auc
    y = pd.Series(1, index=idx)
    assert np.isnan(safe_roc_auc(y, pd.Series(0.6, index=idx)))


def test_running_scores_cumulative_is_count_weighted(idx) -> None:
    """Cumulative must weight by observations, not average the periods —
    otherwise a short period counts as much as a long one."""
    from src.backtest.scoring import running_scores
    rng = np.random.default_rng(11)
    y = pd.Series(rng.integers(0, 2, len(idx)), index=idx)
    p = pd.Series(np.where(y == 1, 0.54, 0.46), index=idx)

    r = running_scores(p, y, period="YE")
    assert len(r) >= 3
    assert r["cum_n"].iloc[-1] == r["n"].sum()
    # Final cumulative log loss equals the pooled mean.
    assert r["cum_log_loss"].iloc[-1] == pytest.approx(log_loss(p, y).mean(), rel=1e-6)


def test_running_scores_exposes_a_one_off_lucky_period(idx) -> None:
    """
    The trajectory diagnostic that motivated this: a signal present in ONE
    period and absent elsewhere must be visible as such, not hidden inside a
    decent pooled average.
    """
    from src.backtest.scoring import running_scores
    rng = np.random.default_rng(12)
    y = pd.Series(rng.integers(0, 2, len(idx)), index=idx)
    p = pd.Series(0.5, index=idx)
    lucky = idx.year == idx.year[0]
    p[lucky] = np.where(y[lucky] == 1, 0.75, 0.25)

    r = running_scores(p, y, period="YE")
    edges = r["log_loss_edge"]
    assert edges.iloc[0] > 0.1          # the lucky year stands out
    assert edges.iloc[1:].abs().max() < 0.01   # the rest are flat


# --- overlapping-returns trap ----------------------------------------------

def test_overlap_warning_fires_for_multibar_horizons() -> None:
    from src.backtest.scoring import overlap_warning
    assert overlap_warning(1) is None
    w = overlap_warning(20)
    assert w and "overlap" in w and "4.5x" in w


def test_overlapping_returns_inflate_significance() -> None:
    """
    Regression test for a real near-miss: COT at a 20-day horizon scored
    t = -3.75 (through Bonferroni, bound for the holdout) with overlapping
    samples, and t = -0.74 without. The inflation must be reproducible here so
    nobody re-learns it the expensive way.
    """
    rng = np.random.default_rng(20)
    n, H = 4000, 20
    idx = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    r1 = pd.Series(rng.normal(0, 0.005, n), index=idx)
    fwd = r1[::-1].rolling(H).sum()[::-1]          # overlapping H-day returns
    feat = pd.Series(rng.normal(size=n), index=idx)  # pure noise

    d = pd.DataFrame({"f": feat, "y": fwd}).dropna()
    overlapping = signed_return_test(d["f"], d["y"])
    non_overlapping = signed_return_test(d["f"].iloc[::H], d["y"].iloc[::H])

    # Both test pure noise, so neither should be "significant" on average, but
    # the overlapping standard error must be materially smaller.
    assert overlapping["std_error"] < non_overlapping["std_error"]
    assert overlapping["n_effective"] > 10 * non_overlapping["n_effective"]


def test_horizon_is_reported_in_the_result() -> None:
    idx = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    rng = np.random.default_rng(21)
    res = signed_return_test(
        pd.Series(rng.normal(size=300), index=idx),
        pd.Series(rng.normal(0, 0.01, 300), index=idx),
        horizon_bars=20,
    )
    assert res["overlap_warning"] is not None

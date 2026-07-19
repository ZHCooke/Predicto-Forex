"""
The holdout must be genuinely hard to touch by accident, and every deliberate
touch must be logged. These tests pin that contract.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest import holdout as H
from src.backtest.holdout import (
    HoldoutViolation,
    PreRegistration,
    bonferroni_threshold,
    evaluate_against_prereg,
    research_only,
    split_research_holdout,
)


@pytest.fixture
def df():
    idx = pd.date_range("2015-01-01", "2026-07-01", freq="D", tz="UTC")
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Redirect pre-registrations and the access log into a temp dir."""
    monkeypatch.setattr(H, "PREREG_DIR", tmp_path / "prereg")
    monkeypatch.setattr(H, "ACCESS_LOG", tmp_path / "holdout_access.log")
    monkeypatch.setattr(H, "LOG_DIR", tmp_path)


def _prereg(hid="h1", predicted=0.6):
    return PreRegistration(
        hypothesis_id=hid, description="daily momentum + carry", symbol="EURUSD",
        timeframe="1d", model="naive_momentum", horizon=1,
        predicted_net_sharpe=predicted,
    )


def test_split_is_disjoint_and_complete(df) -> None:
    research, hold = split_research_holdout(df)
    assert len(research) + len(hold) == len(df)
    assert research.index.max() < hold.index.min()


def test_research_stops_before_the_seal(df) -> None:
    r = research_only(df)
    assert r.index.max() < pd.Timestamp(H.HOLDOUT_START, tz="UTC")


def test_holdout_starts_at_the_seal(df) -> None:
    _, hold = split_research_holdout(df)
    assert hold.index.min() >= pd.Timestamp(H.HOLDOUT_START, tz="UTC")


def test_holdout_is_a_meaningful_fraction(df) -> None:
    research, hold = split_research_holdout(df)
    frac = len(hold) / len(df)
    assert 0.2 < frac < 0.45, f"holdout is {frac:.0%} of the sample"


def test_unlocking_without_prereg_raises(df) -> None:
    with pytest.raises(HoldoutViolation, match="no pre-registration"):
        H.unlock_holdout(df, "never_registered")


def test_unlock_succeeds_after_prereg(df) -> None:
    _prereg().save()
    hold, pr = H.unlock_holdout(df, "h1")
    assert len(hold) > 0
    assert pr.predicted_net_sharpe == 0.6


def test_access_is_logged_and_counted(df) -> None:
    _prereg().save()
    H.unlock_holdout(df, "h1")
    H.unlock_holdout(df, "h1")
    assert H._access_count("h1") == 2
    assert H.ACCESS_LOG.exists()


def test_repeat_access_is_surfaced(df, caplog) -> None:
    _prereg().save()
    H.unlock_holdout(df, "h1")
    with caplog.at_level("WARNING"):
        H.unlock_holdout(df, "h1")
    assert "already been accessed" in caplog.text


def test_prereg_cannot_be_overwritten() -> None:
    _prereg().save()
    with pytest.raises(HoldoutViolation, match="already exists"):
        _prereg(predicted=99.0).save()


def test_confirmation_requires_landing_near_the_prediction() -> None:
    pr = _prereg(predicted=0.6)
    assert evaluate_against_prereg(pr, 0.7)["confirmed"]
    # Wildly exceeding the prediction is ALSO a failure — the model of the
    # world was wrong, which usually means a bug or a regime change.
    assert not evaluate_against_prereg(pr, 3.0)["confirmed"]
    assert not evaluate_against_prereg(pr, -0.4)["confirmed"]


def test_degenerate_result_cannot_be_confirmed() -> None:
    """
    Regression test for the real h001 outcome: Sharpe 0.552 against a predicted
    0.40 is within tolerance, but it traded in 0.89% of bars. That is not a
    pass — it is no test at all.
    """
    pr = _prereg(predicted=0.40, hid="h_degen")
    out = evaluate_against_prereg(pr, 0.552, degeneracy=["in-market only 0.89% of bars"])
    assert out["within_tolerance"], "should still report that the number landed"
    assert not out["confirmed"]
    assert out["verdict"] == "INCONCLUSIVE (degenerate)"


def test_verdicts_are_distinct() -> None:
    pr = _prereg(predicted=0.6, hid="h_verdict")
    assert evaluate_against_prereg(pr, 0.7)["verdict"] == "CONFIRMED"
    assert evaluate_against_prereg(pr, -0.9)["verdict"] == "REFUTED"
    assert evaluate_against_prereg(pr, 0.7, ["barely trades"])["verdict"].startswith("INCONCLUSIVE")


def test_bonferroni_tightens_with_search_effort() -> None:
    assert bonferroni_threshold(1) == pytest.approx(0.05)
    # 33 configurations searched -> need ~99.85% confidence, not 95%.
    assert bonferroni_threshold(33) == pytest.approx(0.05 / 33)
    assert bonferroni_threshold(33) < 0.002

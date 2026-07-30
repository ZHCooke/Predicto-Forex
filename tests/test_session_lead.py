"""
Tests for the cross-asset session-transmission machinery (CLAUDE.md s16).

The load-time functions (session_frame) hit real parquet and are exercised by
the battery run itself; here we lock down the pure logic that lookahead-safety
and the correction depend on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.session_lead import _ols_one_sided, benjamini_hochberg
from src.features.cross_asset import pair_contemporaneous, pair_lead


def _sessions(times_utc: list[str], rets: list[float]) -> pd.DataFrame:
    """Build a session frame: each row is one window with start/end/ret."""
    ts = pd.to_datetime(times_utc, utc=True)
    return pd.DataFrame(
        {"start_utc": ts, "end_utc": ts + pd.Timedelta(hours=4), "ret": rets},
        index=pd.DatetimeIndex([t.date() for t in ts], name="session_date"),
    )


def test_pair_lead_is_forward_only_no_lookahead():
    # Predictor ends each day at 21:00 UTC; target opens next day 07:00 UTC.
    pred = _sessions(["2020-01-06 17:00", "2020-01-07 17:00"], [0.01, -0.02])
    tgt = _sessions(["2020-01-07 07:00", "2020-01-08 07:00"], [0.5, -0.5])
    paired = pair_lead(pred, tgt)

    # Every target must start at or after its predictor ends — the whole point.
    assert (paired["gap_hours"] >= 0).all()
    # Monday predictor pairs with Tuesday target, not same day.
    first = paired.iloc[0]
    assert first["pred_ret"] == 0.01
    assert first["tgt_ret"] == 0.5


def test_pair_lead_bridges_the_weekend():
    # Friday afternoon predictor -> Monday open target (~58h gap), no Sat/Sun.
    pred = _sessions(["2020-01-10 17:00"], [0.03])   # Friday
    tgt = _sessions(["2020-01-13 07:00"], [0.9])     # Monday
    paired = pair_lead(pred, tgt)
    assert len(paired) == 1
    assert 48 < paired.iloc[0]["gap_hours"] < 72


def test_pair_lead_drops_when_no_target_within_window():
    pred = _sessions(["2020-01-06 17:00"], [0.01])
    tgt = _sessions(["2020-02-01 07:00"], [0.5])     # weeks later, beyond tolerance
    assert pair_lead(pred, tgt).empty


def test_pair_contemporaneous_matches_same_session():
    pred = _sessions(["2020-01-06 08:00", "2020-01-07 08:00"], [0.01, 0.02])
    tgt = _sessions(["2020-01-06 08:00", "2020-01-07 08:00"], [0.1, 0.2])
    paired = pair_contemporaneous(pred, tgt)
    assert len(paired) == 2
    assert (paired["gap_hours"] == 0).all()
    assert list(paired["pred_ret"]) == [0.01, 0.02]


def test_ols_one_sided_direction():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=500))
    y = 0.4 * x + rng.normal(size=500)  # genuine positive slope

    right = _ols_one_sided(x, y, sign=+1)
    wrong = _ols_one_sided(x, y, sign=-1)
    assert right["sign_matches"] and right["p_one_sided"] < 0.01
    # Same data tested against the wrong pre-registered direction must NOT pass.
    assert not wrong["sign_matches"] and wrong["p_one_sided"] > 0.99


def test_benjamini_hochberg_step_up():
    # 4 tiny p-values, 6 null-ish. BH at q=0.10 should reject the small ones.
    p = pd.Series([0.001, 0.002, 0.003, 0.004, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    rej = benjamini_hochberg(p, q=0.10)
    assert rej.iloc[:4].all()
    assert not rej.iloc[4:].any()


def test_benjamini_hochberg_all_null():
    p = pd.Series([0.5, 0.6, 0.7, 0.8])
    assert not benjamini_hochberg(p, q=0.10).any()

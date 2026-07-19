"""
The test CLAUDE.md s4.1 mandates: prove empirically that no feature sees the future.

Method: build features on a series, then corrupt everything after a cut point
and rebuild. Every feature value at or before the cut must be bit-identical.
If a feature peeks forward — a centered window, a bfill, a negative shift —
the corrupted future bleeds backwards and the comparison fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import assemble_dataset, build_features, make_target

CUT = 1500


def test_no_feature_depends_on_the_future(bars: pd.DataFrame) -> None:
    original = build_features(bars)

    corrupted_bars = bars.copy()
    # Violently different future: 10x the price level after the cut.
    corrupted_bars.iloc[CUT:] = corrupted_bars.iloc[CUT:] * 10
    corrupted = build_features(corrupted_bars)

    head_orig = original.iloc[:CUT]
    head_corr = corrupted.iloc[:CUT]

    offenders = [
        col for col in head_orig.columns
        if not head_orig[col].equals(head_corr[col])
    ]
    assert not offenders, f"features leaked future information: {offenders}"


@pytest.mark.parametrize("cut", [200, 800, 2500])
def test_lookahead_holds_at_multiple_cut_points(bars: pd.DataFrame, cut: int) -> None:
    original = build_features(bars)
    corrupted_bars = bars.copy()
    corrupted_bars.iloc[cut:] += 0.05
    corrupted = build_features(corrupted_bars)

    pd.testing.assert_frame_equal(original.iloc[:cut], corrupted.iloc[:cut])


def test_prefix_growth_is_stable(bars: pd.DataFrame) -> None:
    """
    Streaming equivalence: features computed on the first k bars must match
    features computed on the full series and then truncated to k. This is the
    property that makes an offline backtest match live execution.
    """
    full = build_features(bars)
    for k in (500, 1200, 2400):
        partial = build_features(bars.iloc[:k])
        pd.testing.assert_frame_equal(partial, full.iloc[:k])


def test_target_is_forward_looking_by_design(bars: pd.DataFrame) -> None:
    """The target MUST see the future — assert it does, so it can't be mistaken
    for a feature that accidentally passes the lookahead test."""
    y = make_target(bars["close"], horizon=1)

    expected = np.log(bars["close"].iloc[1] / bars["close"].iloc[0])
    assert y.iloc[0] == pytest.approx(expected)
    assert np.isnan(y.iloc[-1]), "last row has no future, must be NaN"
    assert not y.name.startswith("f_"), "target must not use the feature prefix"


def test_assembled_dataset_has_no_nans(bars: pd.DataFrame) -> None:
    X, y = assemble_dataset(bars)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert X.index.equals(y.index)
    assert len(X) > 0
    # Warm-up must actually cost rows; if not, a feature isn't using history.
    assert len(X) < len(bars)


def test_features_use_the_f_prefix(bars: pd.DataFrame) -> None:
    X = build_features(bars)
    assert all(c.startswith("f_") for c in X.columns)

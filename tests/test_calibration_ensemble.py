"""
Tests for the two techniques imported from PL-AFL-Module: isotonic calibration
of predicted returns, and ensemble disagreement as a trading filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import BaseModel, RidgeModel
from src.models.calibration import (
    CalibratedModel,
    IsotonicReturnCalibrator,
    calibration_report,
)
from src.models.ensemble import (
    EnsembleModel,
    apply_agreement_filter,
    make_feature_families,
)


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    X = pd.DataFrame(
        {
            "f_a": rng.normal(0, 1, n),
            "f_b": rng.normal(0, 1, n),
            "f_rate_diff": rng.normal(0, 1, n),
            "f_ccy_diff_5": rng.normal(0, 1, n),
        },
        index=idx,
    )
    y = pd.Series(0.001 * X["f_a"] + rng.normal(0, 0.003, n), index=idx, name="y")
    return X, y


# --- calibration ------------------------------------------------------------

def test_calibrator_shrinks_systematic_overprediction() -> None:
    """The core failure it exists to fix: predictions 10x too large."""
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 0.001, 500)
    predicted = actual * 10  # right direction, wildly wrong magnitude

    cal = IsotonicReturnCalibrator().fit(predicted, actual)
    out = cal.transform(predicted)

    assert np.std(out) < np.std(predicted)
    # Calibrated output should sit near the true scale.
    assert np.std(out) == pytest.approx(np.std(actual), rel=0.5)


def test_calibrator_is_monotone() -> None:
    rng = np.random.default_rng(2)
    p = np.sort(rng.normal(0, 1, 300))
    a = p * 0.5 + rng.normal(0, 0.1, 300)
    out = IsotonicReturnCalibrator().fit(p, a).transform(p)
    assert np.all(np.diff(out) >= -1e-12), "calibration must be non-decreasing"


def test_skilless_model_collapses_toward_constant() -> None:
    """
    The useful side effect: if predictions carry no information, isotonic
    flattens, magnitudes collapse, and Kelly will size toward zero. A model
    with nothing to say should stop trading.
    """
    rng = np.random.default_rng(3)
    predicted = rng.normal(0, 1, 600)
    actual = rng.normal(0, 0.001, 600)  # unrelated

    out = IsotonicReturnCalibrator().fit(predicted, actual).transform(predicted)
    assert np.std(out) < 0.2 * np.std(actual) + 1e-9


def test_small_sample_falls_back_to_identity() -> None:
    p = np.array([1.0, 2.0, 3.0])
    cal = IsotonicReturnCalibrator().fit(p, np.array([1.0, 2.0, 3.0]))
    assert not cal.is_fitted
    np.testing.assert_allclose(cal.transform(p), p)


def test_out_of_range_predictions_are_clipped_not_extrapolated() -> None:
    """Extrapolating beyond the calibration range is how oversized positions
    get created; the mapping must stay bounded."""
    rng = np.random.default_rng(4)
    p = rng.normal(0, 1, 400)
    a = p * 0.001
    cal = IsotonicReturnCalibrator().fit(p, a)

    extreme = cal.transform(np.array([1000.0]))
    assert abs(extreme[0]) <= abs(a).max() + 1e-12


def test_calibrated_model_holds_out_a_chronological_slice(xy) -> None:
    X, y = xy
    m = CalibratedModel(RidgeModel, calib_frac=0.3).fit(X, y)
    assert m.calibrator.is_fitted
    preds = m.predict(X)
    assert len(preds) == len(X)
    assert preds.notna().all()


def test_calibrated_model_reduces_magnitude_error(xy) -> None:
    """Calibrated predictions should track realized magnitude better."""
    X, y = xy
    split = int(len(X) * 0.6)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    raw = RidgeModel().fit(Xtr, ytr).predict(Xte)
    cal = CalibratedModel(RidgeModel, calib_frac=0.3).fit(Xtr, ytr).predict(Xte)

    err_raw = abs(raw.std() - yte.std())
    err_cal = abs(cal.std() - yte.std())
    assert err_cal <= err_raw + 1e-9


def test_short_window_skips_calibration_rather_than_crippling_fit() -> None:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    X = pd.DataFrame({"f_a": rng.normal(0, 1, 60)}, index=idx)
    y = pd.Series(rng.normal(0, 0.001, 60), index=idx)

    m = CalibratedModel(RidgeModel, calib_frac=0.3).fit(X, y)
    assert not m.calibrator.is_fitted
    assert m.predict(X).notna().all()


def test_calib_frac_validated() -> None:
    with pytest.raises(ValueError):
        CalibratedModel(RidgeModel, calib_frac=0.0)
    with pytest.raises(ValueError):
        CalibratedModel(RidgeModel, calib_frac=1.0)


def test_calibration_report_flags_overprediction() -> None:
    rng = np.random.default_rng(6)
    actual = pd.Series(rng.normal(0, 0.001, 500))
    predicted = actual * 5
    rep = calibration_report(predicted, actual, n_bins=5)
    assert len(rep) == 5
    assert rep["mean_pred"].is_monotonic_increasing


# --- ensemble ---------------------------------------------------------------

def test_feature_families_split_by_source(xy) -> None:
    X, _ = xy
    fams = make_feature_families(list(X.columns))
    assert len(fams) == 3
    flat = [c for f in fams for c in f]
    assert set(flat) == set(X.columns)
    # Each family must be internally homogeneous.
    assert any("f_rate_diff" in f and len(f) == 1 for f in fams)


def test_ensemble_mean_lies_within_member_range(xy) -> None:
    X, y = xy
    ens = EnsembleModel(RidgeModel, make_feature_families(list(X.columns))).fit(X, y)
    unc = ens.predict_with_uncertainty(X)
    assert (unc["mean"] >= unc["p10"] - 1e-9).all()
    assert (unc["mean"] <= unc["p90"] + 1e-9).all()


def test_agreement_is_bounded_and_meaningful(xy) -> None:
    X, y = xy
    ens = EnsembleModel(RidgeModel, make_feature_families(list(X.columns))).fit(X, y)
    unc = ens.predict_with_uncertainty(X)
    assert unc["agreement"].between(0, 1).all()
    assert unc["agreement"].max() > 0.5


def test_agreement_filter_zeroes_disputed_bars() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    preds = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    agree = pd.Series([1.0, 0.5, 1.0, 0.25, 0.8], index=idx)

    out = apply_agreement_filter(preds, agree, min_agreement=0.75)
    assert out.tolist() == [1.0, 0.0, 3.0, 0.0, 5.0]


def test_agreement_filter_validates_threshold() -> None:
    s = pd.Series([1.0])
    with pytest.raises(ValueError):
        apply_agreement_filter(s, s, min_agreement=0.0)
    with pytest.raises(ValueError):
        apply_agreement_filter(s, s, min_agreement=1.5)


def test_unanimous_ensemble_has_agreement_one() -> None:
    """Members that all point the same way must score 1.0."""
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    X = pd.DataFrame({"f_a": np.linspace(-1, 1, 50), "f_b": np.linspace(-1, 1, 50)}, index=idx)
    y = pd.Series(np.linspace(-1, 1, 50) * 0.001, index=idx)

    class Same(BaseModel):
        name = "same"
        def fit(self, X_, y_): return self
        def predict(self, X_): return pd.Series(1.0, index=X_.index)

    ens = EnsembleModel(Same, [["f_a"], ["f_b"]]).fit(X, y)
    assert ens.predict_with_uncertainty(X)["agreement"].eq(1.0).all()


def test_ensemble_survives_a_failing_member(xy) -> None:
    X, y = xy
    ens = EnsembleModel(RidgeModel, [["f_a"], ["does_not_exist"], ["f_b"]]).fit(X, y)
    assert len(ens.members) == 2
    assert ens.predict(X).notna().all()


def test_ensemble_requires_minimum_members() -> None:
    with pytest.raises(ValueError):
        EnsembleModel(RidgeModel, [["f_a"]])

"""
calibration.py

Isotonic calibration of predicted returns, adapted from
PL-AFL-Module/AFL_0.0.1/models/calibration.py.

WHY WE NEED IT. Kelly sizes positions as mu/sigma^2 — directly proportional to
the predicted return MAGNITUDE. So a model that gets direction roughly right
but magnitude badly wrong will size badly, and we have hard evidence of exactly
that: at daily horizon ridge predicted edges 2.09x breakeven while delivering a
NEGATIVE gross Sharpe. It was confidently wrong, and we sized on the confidence.

WHAT CHANGES FROM THE AFL VERSION. AFL calibrates margin -> win PROBABILITY,
a classification problem, so isotonic is fit against a 0/1 outcome. We are
doing regression: we calibrate predicted return -> EXPECTED ACTUAL return, so
isotonic is fit against the realized continuous return. The monotonicity
assumption is the same and is the reason isotonic is the right tool: we are
willing to assume "a higher prediction should not imply a lower expected
return", but not to assume any particular functional shape.

THE CRITICAL DETAIL — calibrate on data the model did not train on. Fitting the
calibrator on the model's own training predictions teaches it the model's
IN-SAMPLE overconfidence, which is not the overconfidence it will exhibit out
of sample. `CalibratedModel` therefore holds back the most recent slice of each
training window, chronologically (never randomly — this is time series).

A USEFUL SIDE EFFECT. If a model has no real skill, isotonic regression will
flatten toward a constant, predicted magnitudes collapse toward the mean, and
Kelly sizes toward zero. A calibrated model that has nothing to say naturally
stops trading, which is exactly the behaviour we want and the opposite of what
raw magnitudes do.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.models.baseline import BaseModel

log = logging.getLogger(__name__)

MIN_CALIBRATION_ROWS = 50


class IsotonicReturnCalibrator:
    """
    Monotone map from predicted return to expected realized return.

    Falls back to the identity when there is too little data to fit, so a short
    window degrades to "uncalibrated" rather than to something arbitrary.
    """

    def __init__(self, min_rows: int = MIN_CALIBRATION_ROWS):
        self.min_rows = min_rows
        self._ir: IsotonicRegression | None = None
        self.n_train = 0

    def fit(self, predicted: np.ndarray, actual: np.ndarray) -> "IsotonicReturnCalibrator":
        p = np.asarray(predicted, dtype=float).ravel()
        a = np.asarray(actual, dtype=float).ravel()
        ok = np.isfinite(p) & np.isfinite(a)
        p, a = p[ok], a[ok]

        if len(p) < self.min_rows:
            log.warning(
                "calibration set has %d rows (< %d); falling back to identity",
                len(p), self.min_rows,
            )
            return self

        # increasing=True encodes the one assumption we are willing to make.
        # out_of_bounds="clip" keeps predictions beyond the calibration range
        # bounded rather than extrapolating a trend we never observed —
        # extrapolation is precisely where oversized positions come from.
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        ir.fit(p, a)
        self._ir = ir
        self.n_train = len(p)
        return self

    def transform(self, predicted: np.ndarray) -> np.ndarray:
        p = np.asarray(predicted, dtype=float).ravel()
        if self._ir is None:
            return p
        return self._ir.predict(p)

    @property
    def is_fitted(self) -> bool:
        return self._ir is not None


class CalibratedModel(BaseModel):
    """
    Wraps any BaseModel so its predictions pass through an isotonic calibrator.

    The training window is split chronologically: the model fits on the earlier
    portion, the calibrator on the later held-out portion. Both are strictly
    inside the fold's training data, so no test information is touched.
    """

    def __init__(self, base_factory, calib_frac: float = 0.3, name: str | None = None):
        if not 0 < calib_frac < 1:
            raise ValueError("calib_frac must be in (0, 1)")
        self.base_factory = base_factory
        self.calib_frac = calib_frac
        self._base: BaseModel | None = None
        self.calibrator = IsotonicReturnCalibrator()
        self.name = name or "calibrated"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CalibratedModel":
        n = len(X)
        n_calib = int(n * self.calib_frac)

        if n_calib < MIN_CALIBRATION_ROWS:
            # Too little data to spare: fit on everything and skip calibration
            # rather than crippling the model with a tiny training slice.
            log.warning("training window %d too short to hold out a calibration set", n)
            self._base = self.base_factory().fit(X, y)
            return self

        # Chronological split — the calibration slice must be the LATEST data,
        # mirroring how the model will be used (fit on the past, applied to the
        # near future). A random split would leak adjacent bars across the
        # boundary and understate the model's true overconfidence.
        X_fit, y_fit = X.iloc[: n - n_calib], y.iloc[: n - n_calib]
        X_cal, y_cal = X.iloc[n - n_calib :], y.iloc[n - n_calib :]

        self._base = self.base_factory().fit(X_fit, y_fit)
        raw = self._base.predict(X_cal)
        self.calibrator.fit(raw.to_numpy(), y_cal.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._base is None:
            raise RuntimeError("fit() must be called before predict()")
        raw = self._base.predict(X)
        return pd.Series(
            self.calibrator.transform(raw.to_numpy()), index=X.index, name=self.name
        )


def calibration_report(predicted: pd.Series, actual: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """
    Bucket predictions and compare mean predicted vs mean realized return.

    A well-calibrated model tracks the diagonal. The usual failure is a
    `ratio` far above 1 in the extreme buckets — predicting big moves that do
    not materialise, which is what drives oversized Kelly positions.
    """
    df = pd.DataFrame({"pred": predicted, "actual": actual}).dropna()
    if df.empty:
        raise ValueError("no overlapping predictions and outcomes")

    df["bucket"] = pd.qcut(df["pred"], n_bins, labels=False, duplicates="drop")
    out = df.groupby("bucket").agg(
        n=("pred", "size"),
        mean_pred=("pred", "mean"),
        mean_actual=("actual", "mean"),
    )
    out["ratio"] = out["mean_pred"] / out["mean_actual"].replace(0, np.nan)
    return out

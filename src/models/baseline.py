"""
baseline.py

Benchmarks that any real model must beat before it earns further attention
(CLAUDE.md s5: "strong simple baselines before anything complex").

Every model exposes the same tiny interface so the backtest engine can swap
them freely:

    fit(X, y)            -> self
    predict(X)           -> pd.Series of predicted forward returns

Predictions are in the same units as the target (log return over the horizon),
which is what the Kelly sizer expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series: ...


class BuyAndHold(BaseModel):
    """Always long. The benchmark a directional FX strategy must beat."""

    name = "buy_and_hold"

    def __init__(self, size: float = 1.0):
        self.size = size

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BuyAndHold":
        # Scale of the constant prediction doesn't matter for direction, but
        # matching training-set magnitude keeps Kelly sizing comparable.
        self._value = float(abs(y).mean()) * self.size
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._value, index=X.index, name=self.name)


class RandomSignal(BaseModel):
    """
    Coin-flip direction with the training set's typical magnitude. The null
    hypothesis: after costs this should lose money. If it doesn't, the cost
    model is wrong.
    """

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomSignal":
        self._scale = float(abs(y).mean())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        signs = rng.choice([-1.0, 1.0], size=len(X))
        return pd.Series(signs * self._scale, index=X.index, name=self.name)


class NaiveMomentum(BaseModel):
    """
    Predict that the trailing momentum feature persists. The classic FX
    baseline and a surprisingly hard one to beat net of costs.
    """

    name = "naive_momentum"

    def __init__(self, feature: str = "f_mom_12"):
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveMomentum":
        if self.feature not in X.columns:
            raise KeyError(f"{self.feature!r} not in features: {list(X.columns)[:8]}...")
        # Regress the target on the single momentum feature to get the sign and
        # scale right, rather than assuming momentum continues one-for-one.
        x = X[self.feature]
        self._beta = float(np.cov(x, y)[0, 1] / np.var(x)) if np.var(x) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (X[self.feature] * self._beta).rename(self.name)


class MeanReversion(BaseModel):
    """Fade the trailing z-score. The natural counterpart to NaiveMomentum."""

    name = "mean_reversion"

    def __init__(self, feature: str = "f_z_close_96"):
        self.feature = feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanReversion":
        if self.feature not in X.columns:
            raise KeyError(f"{self.feature!r} not in features")
        x = X[self.feature]
        self._beta = float(np.cov(x, y)[0, 1] / np.var(x)) if np.var(x) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (X[self.feature] * self._beta).rename(self.name)


class RidgeModel(BaseModel):
    """
    Linear model over all features. Not a baseline exactly — the simplest
    thing that could actually work, and the bar a tree/NN model must clear.
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Scaler is fit on training data only — refit per fold by the engine,
        # so no test-set statistics leak into the transform.
        self._pipe = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._pipe.fit(X.to_numpy(), y.to_numpy())
        self._columns = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self._pipe.predict(X[self._columns].to_numpy()), index=X.index, name=self.name
        )


BASELINES: dict[str, type[BaseModel]] = {
    m.name: m for m in (BuyAndHold, RandomSignal, NaiveMomentum, MeanReversion, RidgeModel)
}

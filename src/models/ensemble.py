"""
ensemble.py

Ensemble over feature subsets with a disagreement signal, adapted from
PL-AFL-Module/AFL_0.0.1/models/ensemble.py.

TWO REASONS THIS IS WORTH IMPORTING.

1. Per-prediction uncertainty. CLAUDE.md s4.6 requires uncertainty on every
   result, and until now we have only had it in aggregate (bootstrap CIs on the
   whole return series). An ensemble gives a spread on EVERY bar.

2. Disagreement as a trading filter — the more important one. Costs, not
   signal, have been the binding constraint throughout this project. Trading
   only when independently-built models agree cuts the number of trades
   directly, which attacks the constraint from the right side. A single model's
   confidence is nearly worthless as a filter (it is confident and wrong all the
   time — see the ridge case that predicted 2.09x breakeven with negative gross
   Sharpe). Agreement between models built on DIFFERENT information is a much
   harder thing to fake.

Members are trained on different feature subsets rather than on bootstrap
resamples, because our features fall into natural families (price-derived,
carry/rates, cross-pair strength) and disagreement ACROSS families is the
signal we actually care about: it says the evidence sources conflict.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.baseline import BaseModel

log = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Trains one member per feature subset and averages their predictions.

    `predict` returns the mean, so it is a drop-in BaseModel.
    `predict_with_uncertainty` additionally returns the spread and an
    agreement fraction, which the engine uses to filter.
    """

    def __init__(
        self,
        base_factory,
        feature_sets: list[list[str]],
        name: str = "ensemble",
        min_members: int = 2,
    ):
        if len(feature_sets) < min_members:
            raise ValueError(f"need at least {min_members} feature sets")
        self.base_factory = base_factory
        self.feature_sets = [list(fs) for fs in feature_sets]
        self.name = name
        self.members: list[BaseModel] = []
        self._active: list[list[str]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel":
        self.members, self._active = [], []
        for i, cols in enumerate(self.feature_sets):
            present = [c for c in cols if c in X.columns]
            if not present:
                log.warning("member %d has no usable features, skipping", i)
                continue
            try:
                self.members.append(self.base_factory().fit(X[present], y))
                self._active.append(present)
            except Exception as exc:  # noqa: BLE001 - one bad member must not kill the ensemble
                log.warning("member %d failed to fit (%s), skipping", i, exc)

        if not self.members:
            raise RuntimeError("no ensemble member could be fitted")
        return self

    def _member_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                i: m.predict(X[cols]).to_numpy()
                for i, (m, cols) in enumerate(zip(self.members, self._active))
            },
            index=X.index,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self._member_predictions(X).mean(axis=1).rename(self.name)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns per-bar: mean, std, p10, p90, and `agreement`.

        `agreement` is the fraction of members whose predicted DIRECTION matches
        the ensemble mean — 1.0 is unanimous, 0.5 is a coin flip. Direction
        agreement is the right measure here because direction is what we
        actually trade; two members can differ wildly on magnitude while both
        saying "up", and that is a usable signal.
        """
        preds = self._member_predictions(X)
        mean = preds.mean(axis=1)

        sign_mean = np.sign(mean)
        agreement = preds.apply(np.sign).eq(sign_mean, axis=0).mean(axis=1)
        # An exactly-zero mean has no direction to agree with; treat as no
        # consensus rather than as unanimous.
        agreement = agreement.where(sign_mean != 0, 0.0)

        return pd.DataFrame(
            {
                "mean": mean,
                "std": preds.std(axis=1),
                "p10": preds.quantile(0.1, axis=1),
                "p90": preds.quantile(0.9, axis=1),
                "agreement": agreement,
                "n_members": len(self.members),
            }
        )


def apply_agreement_filter(
    predictions: pd.Series, agreement: pd.Series, min_agreement: float = 0.75
) -> pd.Series:
    """
    Zero out predictions the ensemble does not agree on.

    Applied BEFORE Kelly sizing, so a disputed bar produces no position and
    therefore no trade and no cost. With four members, min_agreement=0.75 means
    at least three of four must point the same way.
    """
    if not 0 < min_agreement <= 1:
        raise ValueError("min_agreement must be in (0, 1]")
    aligned = agreement.reindex(predictions.index).fillna(0.0)
    return predictions.where(aligned >= min_agreement, 0.0)


def make_feature_families(columns: list[str]) -> list[list[str]]:
    """
    Split a feature matrix into its natural families.

    Disagreement between families is the meaningful kind: it says the price
    action, the rates, and the cross-currency evidence are pointing different
    ways. A split into arbitrary random subsets would mostly measure sampling
    noise instead.
    """
    families = {
        "price": [c for c in columns if c.startswith("f_") and not c.startswith(("f_rate", "f_ccy", "f_dollar"))],
        "carry": [c for c in columns if c.startswith("f_rate")],
        "strength": [c for c in columns if c.startswith(("f_ccy", "f_dollar"))],
    }
    return [cols for cols in families.values() if cols]

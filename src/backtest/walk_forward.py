"""
walk_forward.py

Rolling / expanding window train-test splitting (CLAUDE.md s4.2). A single
train/test split is not acceptable in this project — we report performance
ACROSS windows, including its variance, not a single mean.

Embargo: a gap of `embargo` bars is left between the end of train and the
start of test. With a forward-looking target of horizon h, the last h training
labels are computed from bars that fall inside the test window; without an
embargo of at least h, that leaks. Default embargo should be >= your target
horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """One walk-forward fold, as positional index arrays."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __repr__(self) -> str:
        return (
            f"Split(fold={self.fold}, train={len(self.train_idx)}, "
            f"test={len(self.test_idx)})"
        )


class WalkForwardSplitter:
    """
    Generate sequential train/test folds over a time-ordered dataset.

    mode="expanding": train window grows from the start each fold (more data,
        assumes the relationship is stable).
    mode="rolling": train window is a fixed length that slides forward (adapts
        to regime change, less data per fit).
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        embargo: int = 1,
        mode: Literal["expanding", "rolling"] = "expanding",
        step: int | None = None,
    ):
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")

        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo
        self.mode = mode
        # Non-overlapping test windows by default: every bar is tested once.
        self.step = step or test_size

    def n_splits(self, n_samples: int) -> int:
        span = self.train_size + self.embargo + self.test_size
        if n_samples < span:
            return 0
        return 1 + (n_samples - span) // self.step

    def split(self, X: pd.DataFrame | pd.Series | np.ndarray) -> Iterator[Split]:
        n = len(X)
        span = self.train_size + self.embargo + self.test_size
        if n < span:
            raise ValueError(
                f"need at least {span} samples for one fold, got {n}. "
                "Shorten train_size/test_size or pull more history."
            )

        for fold, start in enumerate(range(0, n - span + 1, self.step)):
            train_end = start + self.train_size
            test_start = train_end + self.embargo
            test_end = test_start + self.test_size

            train_start = start if self.mode == "rolling" else 0
            yield Split(
                fold=fold,
                train_idx=np.arange(train_start, train_end),
                test_idx=np.arange(test_start, test_end),
            )


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold metrics into mean / std / min / max.

    The std column is the point of walk-forward: a strategy with a good mean
    Sharpe and huge across-fold variance has not been validated, it has been
    averaged. Read the spread before the mean.
    """
    return fold_metrics.agg(["mean", "std", "min", "max"]).T

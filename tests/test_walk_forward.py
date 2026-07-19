"""Splitter invariants: no fold may train on data at or after its own test window."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtest.walk_forward import WalkForwardSplitter


@pytest.fixture
def X():
    import pandas as pd
    return pd.DataFrame({"a": np.arange(1000)})


def test_train_always_precedes_test(X) -> None:
    splitter = WalkForwardSplitter(train_size=200, test_size=50, embargo=5)
    folds = list(splitter.split(X))
    assert folds

    for f in folds:
        assert f.train_idx.max() < f.test_idx.min(), "train overlaps test"


def test_embargo_gap_is_respected(X) -> None:
    embargo = 10
    splitter = WalkForwardSplitter(train_size=200, test_size=50, embargo=embargo)
    for f in splitter.split(X):
        gap = f.test_idx.min() - f.train_idx.max() - 1
        assert gap == embargo


def test_expanding_train_grows_rolling_does_not(X) -> None:
    expanding = list(WalkForwardSplitter(200, 50, mode="expanding").split(X))
    rolling = list(WalkForwardSplitter(200, 50, mode="rolling").split(X))

    exp_sizes = [len(f.train_idx) for f in expanding]
    assert exp_sizes == sorted(exp_sizes) and exp_sizes[-1] > exp_sizes[0]

    assert {len(f.train_idx) for f in rolling} == {200}
    # Rolling windows must actually move forward.
    assert rolling[1].train_idx.min() > rolling[0].train_idx.min()


def test_test_windows_do_not_overlap(X) -> None:
    folds = list(WalkForwardSplitter(200, 50).split(X))
    seen: set[int] = set()
    for f in folds:
        assert not seen & set(f.test_idx.tolist()), "test windows overlap"
        seen |= set(f.test_idx.tolist())


def test_test_windows_advance_in_time(X) -> None:
    folds = list(WalkForwardSplitter(200, 50).split(X))
    starts = [f.test_idx.min() for f in folds]
    assert starts == sorted(starts)


def test_n_splits_matches_actual(X) -> None:
    splitter = WalkForwardSplitter(200, 50, embargo=5)
    assert splitter.n_splits(len(X)) == len(list(splitter.split(X)))


def test_too_short_dataset_raises() -> None:
    import pandas as pd
    splitter = WalkForwardSplitter(200, 50)
    assert splitter.n_splits(10) == 0
    with pytest.raises(ValueError, match="at least"):
        list(splitter.split(pd.DataFrame({"a": np.arange(10)})))

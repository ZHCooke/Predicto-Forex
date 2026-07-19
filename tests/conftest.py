"""Shared fixtures: synthetic OHLCV bars so tests need no network or stored data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 3000, seed: int = 0, freq: str = "15min") -> pd.DataFrame:
    """Random-walk OHLCV bars on a tz-aware UTC index, shaped like real output."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC", name="timestamp")

    steps = rng.normal(0, 0.0004, size=n)
    close = 1.10 * np.exp(np.cumsum(steps))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.0003, size=n))

    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.uniform(100, 1000, size=n),
        },
        index=idx,
    )


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def small_bars() -> pd.DataFrame:
    return make_bars(n=600, seed=7)

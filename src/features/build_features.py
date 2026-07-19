"""
build_features.py

Feature engineering on OHLCV bars.

LOOKAHEAD CONTRACT (CLAUDE.md s4.1) — every function here obeys:
    feature value at row t may only use bar data from rows <= t.

That means: rolling windows are backward-looking and never centered, no
`.shift(-n)`, no `bfill`. The one deliberate exception is `make_target()`,
which is forward-looking BY DEFINITION — it is the label, not a feature, and
is named/segregated so it can never be mistaken for one.

`tests/test_no_lookahead.py` enforces this empirically: it perturbs the future
of a series and asserts no feature column changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bars per year, used to annualize realized volatility.
BARS_PER_YEAR = {
    "1min": 372_000, "5min": 74_400, "15min": 24_800, "30min": 12_400,
    "1h": 6_200, "4h": 1_550, "1d": 260,
}

FEATURE_PREFIX = "f_"


def log_returns(close: pd.Series) -> pd.Series:
    """Bar-over-bar log return. Uses t and t-1 only."""
    return np.log(close / close.shift(1))


def realized_vol(returns: pd.Series, window: int, timeframe: str) -> pd.Series:
    """Annualized rolling standard deviation of returns over a trailing window."""
    scale = np.sqrt(BARS_PER_YEAR.get(timeframe, 252))
    return returns.rolling(window).std() * scale


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder-style RSI on a trailing window, scaled to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).div(100)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score — the mean-reversion workhorse."""
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range, normalized by close so it's comparable across pairs."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean() / df["close"]


def build_features(
    df: pd.DataFrame,
    timeframe: str = "15min",
    momentum_windows: tuple[int, ...] = (4, 12, 48),
    vol_windows: tuple[int, ...] = (24, 96),
) -> pd.DataFrame:
    """
    Build the feature matrix from OHLCV bars.

    Returns a frame indexed like `df`, with every column prefixed `f_`.
    Warm-up rows containing NaN are NOT dropped here — the backtest splitter
    decides what to do with them, so callers can see how much history each
    feature needs.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    ret = log_returns(df["close"])
    out = pd.DataFrame(index=df.index)

    out["f_ret_1"] = ret

    for w in momentum_windows:
        # Cumulative return over the trailing w bars.
        out[f"f_mom_{w}"] = ret.rolling(w).sum()
        # Where price sits inside its own trailing range: 0 = low, 1 = high.
        lo = df["low"].rolling(w).min()
        hi = df["high"].rolling(w).max()
        out[f"f_pos_in_range_{w}"] = (df["close"] - lo) / (hi - lo).replace(0, np.nan)

    for w in vol_windows:
        out[f"f_vol_{w}"] = realized_vol(ret, w, timeframe)
        out[f"f_z_close_{w}"] = zscore(df["close"], w)

    out["f_rsi_14"] = rsi(df["close"], 14)
    out["f_atr_14"] = atr(df, 14)

    # Session/seasonality — FX behaviour is strongly time-of-day dependent.
    # Encoded cyclically so 23:00 and 00:00 are adjacent, not maximally distant.
    hour = df.index.hour + df.index.minute / 60
    out["f_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["f_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["f_dayofweek"] = df.index.dayofweek / 4.0

    assert all(c.startswith(FEATURE_PREFIX) for c in out.columns)
    return out


def make_target(close: pd.Series, horizon: int = 1) -> pd.Series:
    """
    FORWARD-LOOKING BY DESIGN. The label: log return over the next `horizon`
    bars. Never feed this, or anything derived from it, back in as a feature.

    Row t holds the return from t to t+horizon, so the last `horizon` rows are
    NaN and must be dropped before training.
    """
    return np.log(close.shift(-horizon) / close).rename(f"y_fwd_ret_{horizon}")


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Aggregate bars to a coarser timeframe (e.g. "15min" -> "4h" or "1D").

    Right-closed / right-labelled: a bar stamped 16:00 covers (12:00, 16:00],
    so its close is the price AT 16:00 and no part of the bar lies in the
    future of its own timestamp. Getting this wrong shifts every bar half a
    period into the future and manufactures lookahead.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, closed="right", label="right").agg(agg)
    return out.dropna(subset=["close"])


def assemble_dataset(
    df: pd.DataFrame,
    timeframe: str = "15min",
    horizon: int = 1,
    symbol: str | None = None,
    with_macro: bool = False,
    macro_lag_days: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Features + aligned target, with warm-up and tail NaN rows dropped.

    `with_macro=True` appends carry/rates features from FRED. These are the
    only features here that are exogenous to the price path — see
    src/ingest/fetch_fred.py for the causality contract they obey.
    """
    X = build_features(df, timeframe=timeframe)

    if with_macro:
        if symbol is None:
            raise ValueError("symbol is required when with_macro=True")
        # Imported lazily so the price-only path never needs network or cache.
        from src.ingest.fetch_fred import build_rate_features

        macro = build_rate_features(X.index, symbol=symbol, lag_days=macro_lag_days)
        X = X.join(macro)

    y = make_target(df["close"], horizon=horizon)
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]

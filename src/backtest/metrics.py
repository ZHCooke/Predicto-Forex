"""
metrics.py

Performance statistics, plus block-bootstrap confidence intervals.

CLAUDE.md s4.6 requires uncertainty on every reported number, so `summarize`
returns point estimates and `bootstrap_ci` gives the interval. Block bootstrap
(not iid) because FX returns are autocorrelated in volatility — resampling
single bars would destroy that structure and understate the interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class PerformanceSummary:
    n_bars: int
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    turnover: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compounded equity from log-ish simple returns, starting at 1.0."""
    return (1 + returns.fillna(0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve, as a negative number."""
    eq = equity_curve(returns)
    return float((eq / eq.cummax() - 1).min())


def sharpe(returns: pd.Series, bars_per_year: int) -> float:
    sd = returns.std()
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(bars_per_year))


def sortino(returns: pd.Series, bars_per_year: int) -> float:
    """Like Sharpe but only penalizes downside deviation."""
    downside = returns[returns < 0]
    dd = downside.std()
    if not np.isfinite(dd) or dd == 0:
        return float("nan")
    return float(returns.mean() / dd * np.sqrt(bars_per_year))


def summarize(
    returns: pd.Series, bars_per_year: int, positions: pd.Series | None = None
) -> PerformanceSummary:
    """Point-estimate performance stats for one return stream."""
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("cannot summarize an empty return series")

    years = len(returns) / bars_per_year
    total = float(equity_curve(returns).iloc[-1] - 1)
    ann_ret = float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan")
    mdd = max_drawdown(returns)

    if positions is None:
        turn = float("nan")
    else:
        pos = positions.reindex(returns.index).fillna(0.0)
        turn = float(pos.diff().abs().fillna(pos.abs()).sum() / years)

    return PerformanceSummary(
        n_bars=len(returns),
        total_return=total,
        ann_return=ann_ret,
        ann_vol=float(returns.std() * np.sqrt(bars_per_year)),
        sharpe=sharpe(returns, bars_per_year),
        sortino=sortino(returns, bars_per_year),
        max_drawdown=mdd,
        calmar=float(ann_ret / abs(mdd)) if mdd < 0 else float("nan"),
        hit_rate=float((returns > 0).mean()),
        turnover=turn,
    )


def bootstrap_ci(
    returns: pd.Series,
    bars_per_year: int,
    statistic=sharpe,
    block_size: int = 96,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """
    Circular block bootstrap CI for a statistic of the return series.

    Blocks preserve local autocorrelation and vol clustering; `block_size`
    should be comfortably longer than the strategy's holding period.
    Returns (lower, upper, full resampled distribution).
    """
    returns = returns.dropna()
    n = len(returns)
    if n < block_size:
        raise ValueError(f"series length {n} shorter than block_size {block_size}")

    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    n_blocks = int(np.ceil(n / block_size))

    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, n, size=n_blocks)
        # Wrap around the end so every observation is equally likely to be drawn.
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel() % n
        sample = pd.Series(values[idx[:n]])
        stats[i] = statistic(sample, bars_per_year)

    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), stats


def format_report(
    gross: PerformanceSummary, net: PerformanceSummary, ci: tuple[float, float] | None = None
) -> str:
    """Gross and net side by side — never report one without the other."""
    lines = [
        f"{'metric':<16}{'gross':>12}{'net':>12}",
        "-" * 40,
    ]
    for key in gross.as_dict():
        g, n = getattr(gross, key), getattr(net, key)
        fmt = (lambda v: f"{v:>12.4f}") if isinstance(g, float) else (lambda v: f"{v:>12}")
        lines.append(f"{key:<16}{fmt(g)}{fmt(n)}")
    if ci is not None:
        lines.append("-" * 40)
        lines.append(f"net Sharpe 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    return "\n".join(lines)

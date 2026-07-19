"""
holdout.py

A true out-of-sample period that must never be touched during search.

WHY THIS EXISTS: by the end of session 4 we had run ~33 backtest
configurations across the full 2015-2026 sample. At 95% confidence roughly 1.7
of those are expected to clear zero by chance alone, and indeed exactly one
did (daily momentum, CI lower bound 0.102). Because every configuration saw
every bar, there was no clean data left to adjudicate it. That is the
definition of a contaminated sample.

THE RULE:
    Research period  (< HOLDOUT_START): search freely, as many configs as you like.
    Holdout period  (>= HOLDOUT_START): locked. One look per pre-registered
                                        hypothesis, and the look is logged.

Pre-registration is not bureaucracy — it is what converts "we found something"
into "we predicted something and were right". Writing the hypothesis down
BEFORE seeing the holdout is the only thing that makes the resulting p-value
mean what it claims to mean.

Every holdout access is appended to logs/holdout_access.log. If that file shows
ten evaluations of slightly different variants, the holdout is burnt and the
honest move is to say so rather than to report the best of the ten.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import LOG_DIR, PROJECT_ROOT

log = logging.getLogger(__name__)

# Everything from this date forward is sealed. Chosen to leave ~3.5 years of
# holdout against ~8 years of research data.
HOLDOUT_START = date(2023, 1, 1)

PREREG_DIR = PROJECT_ROOT / "preregistrations"
ACCESS_LOG = LOG_DIR / "holdout_access.log"


class HoldoutViolation(RuntimeError):
    """Raised on any attempt to read the holdout without a pre-registration."""


@dataclass
class PreRegistration:
    """
    A hypothesis committed to BEFORE the holdout is evaluated.

    `predicted_net_sharpe` is deliberately required: a hypothesis that does not
    state what it expects cannot be wrong, and a claim that cannot be wrong is
    not a result. State the number you expect to see.
    """

    hypothesis_id: str
    description: str
    symbol: str
    timeframe: str
    model: str
    horizon: int
    predicted_net_sharpe: float
    tolerance: float = 0.5
    rationale: str = ""
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def path(self) -> Path:
        return PREREG_DIR / f"{self.hypothesis_id}.json"

    def save(self) -> Path:
        if self.path.exists():
            raise HoldoutViolation(
                f"pre-registration {self.hypothesis_id!r} already exists. Editing a "
                "hypothesis after the fact defeats the point — pick a new id."
            )
        PREREG_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        log.info("pre-registered %s -> %s", self.hypothesis_id, self.path)
        return self.path

    @classmethod
    def load(cls, hypothesis_id: str) -> "PreRegistration":
        path = PREREG_DIR / f"{hypothesis_id}.json"
        if not path.exists():
            raise HoldoutViolation(
                f"no pre-registration found for {hypothesis_id!r}. Write down what you "
                "expect BEFORE looking at the holdout — that is the whole mechanism."
            )
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def split_research_holdout(
    df: pd.DataFrame | pd.Series, holdout_start: date = HOLDOUT_START
) -> tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """Split on the seal date. Returns (research, holdout)."""
    cut = pd.Timestamp(holdout_start, tz="UTC")
    return df.loc[df.index < cut], df.loc[df.index >= cut]


def research_only(df: pd.DataFrame | pd.Series, holdout_start: date = HOLDOUT_START):
    """
    The default accessor for all exploratory work. Use this everywhere during
    search so the holdout cannot leak in by accident.
    """
    research, _ = split_research_holdout(df, holdout_start)
    return research


def unlock_holdout(
    df: pd.DataFrame | pd.Series,
    hypothesis_id: str,
    holdout_start: date = HOLDOUT_START,
):
    """
    Read the holdout for ONE pre-registered hypothesis, and log the access.

    Raises HoldoutViolation if no pre-registration exists. The access log is
    append-only and is the audit trail — if it shows many looks, the holdout
    is spent and any subsequent "result" is just the best of N.
    """
    prereg = PreRegistration.load(hypothesis_id)
    _, holdout = split_research_holdout(df, holdout_start)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prior = _access_count(hypothesis_id)
    with ACCESS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "hypothesis_id": hypothesis_id,
                    "rows": len(holdout),
                    "span": [str(holdout.index.min()), str(holdout.index.max())],
                    "prior_accesses": prior,
                }
            )
            + "\n"
        )

    if prior:
        log.warning(
            "holdout has already been accessed %d time(s) for %r. Each extra look "
            "inflates the false-positive rate; report this count with any result.",
            prior, hypothesis_id,
        )
    log.info("holdout unlocked for %r: %d rows", hypothesis_id, len(holdout))
    return holdout, prereg


def _access_count(hypothesis_id: str) -> int:
    if not ACCESS_LOG.exists():
        return 0
    return sum(
        1
        for line in ACCESS_LOG.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("hypothesis_id") == hypothesis_id
    )


def evaluate_against_prereg(
    prereg: PreRegistration,
    observed_net_sharpe: float,
    degeneracy: list[str] | None = None,
) -> dict:
    """
    Compare the holdout outcome to what was predicted. Confirmation requires
    landing within `tolerance` of the prediction — not merely being positive.
    A strategy that predicted 0.6 and delivered 3.0 has also failed: the model
    of the world was wrong, and that usually means a bug or a regime shift.

    `degeneracy` is REQUIRED to be clean for confirmation. This is not
    belt-and-braces — on the very first use of this module, h001 landed at
    Sharpe 0.552 against a predicted 0.40 (nominally a pass) while trading in
    only 0.89% of bars for a total return of 0.7% over 3.5 years. A Sharpe
    computed on almost no exposure is not evidence of anything, and a
    pre-registration framework that rubber-stamps it is worse than none.
    """
    delta = observed_net_sharpe - prereg.predicted_net_sharpe
    within = abs(delta) <= prereg.tolerance and observed_net_sharpe > 0
    degeneracy = degeneracy or []

    return {
        "hypothesis_id": prereg.hypothesis_id,
        "predicted": prereg.predicted_net_sharpe,
        "observed": observed_net_sharpe,
        "delta": delta,
        "tolerance": prereg.tolerance,
        "within_tolerance": within,
        "degeneracy": degeneracy,
        # Both conditions required. A degenerate result is INCONCLUSIVE, not a
        # pass and not a refutation — there was no real out-of-sample test.
        "confirmed": within and not degeneracy,
        "verdict": (
            "CONFIRMED" if (within and not degeneracy)
            else "INCONCLUSIVE (degenerate)" if degeneracy
            else "REFUTED"
        ),
        "accesses": _access_count(prereg.hypothesis_id),
    }


def bonferroni_threshold(n_configurations_tested: int, alpha: float = 0.05) -> float:
    """
    Corrected per-test alpha after searching `n` configurations.

    With n=33 and alpha=0.05 this is 0.0015, i.e. a result needs a ~99.85%
    interval excluding zero, not a 95% one. This is why the session-4 momentum
    candidate does not survive.
    """
    return alpha / max(1, n_configurations_tested)

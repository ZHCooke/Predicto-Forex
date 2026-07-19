"""Project paths and config loading. Everything else imports paths from here."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "instruments.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    timeframe: str
    offer_side: str
    start: date
    end: date
    pip: float
    typical_spread_pips: float

    @property
    def spread_price(self) -> float:
        """Typical spread expressed in price units rather than pips."""
        return self.typical_spread_pips * self.pip


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def load_instruments(
    path: Path | None = None, enabled_only: bool = True
) -> dict[str, InstrumentConfig]:
    """Read config/instruments.yaml, folding `defaults` into each instrument."""
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})

    out: dict[str, InstrumentConfig] = {}
    for symbol, spec in (raw.get("instruments") or {}).items():
        merged = {**defaults, **(spec or {})}
        if enabled_only and not merged.get("enabled", True):
            continue
        out[symbol] = InstrumentConfig(
            symbol=symbol,
            timeframe=merged["timeframe"],
            offer_side=merged.get("offer_side", "bid"),
            start=_as_date(merged["start"]),
            end=_as_date(merged["end"]),
            pip=float(merged["pip"]),
            typical_spread_pips=float(merged["typical_spread_pips"]),
        )
    return out

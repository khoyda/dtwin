"""Load and access the project configuration (config/config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/canola_dt/config.py -> root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    """Thin wrapper around the parsed YAML config with convenience accessors."""

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = PROJECT_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a path defined under the ``paths`` section against the root."""
        rel = self.raw["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def agronomy(self) -> dict[str, Any]:
        return self.raw["agronomy"]

    @property
    def water_balance(self) -> dict[str, Any]:
        return self.raw["water_balance"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]


def load_config(path: str | Path | None = None) -> Config:
    """Read ``config.yaml`` (or a custom path) into a :class:`Config`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=PROJECT_ROOT)

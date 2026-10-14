"""Project-wide configuration (API keys, paths)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

__all__ = ["Settings", "PROJECT_ROOT"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration loaded from environment variables / ``.env``."""

    fred_api_key: str | None = None
    eia_api_key: str | None = None
    data_dir: Path = PROJECT_ROOT / "data"

    @classmethod
    def load(cls) -> Settings:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            fred_api_key=os.getenv("FRED_API_KEY") or None,
            eia_api_key=os.getenv("EIA_API_KEY") or None,
        )

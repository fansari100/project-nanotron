"""Runtime configuration loaded from env + the project's TOML files.

The TOML files in `config/` are the source of truth for risk limits and
strategy params; the control plane mounts them read-only and exposes
them through typed endpoints.  Mutations go through the same models
and are atomically rewritten — never edited in place — so the running
engine always reads a consistent file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_config_root() -> Path:
    """Climb out of `control_plane/...` to the repo root if running from source."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "config"
        if candidate.is_dir() and (candidate / "risk.toml").exists():
            return candidate
    return Path("config")


class Settings(BaseSettings):
    """Process-wide configuration.

    All fields are overridable via env vars prefixed with ``NANOTRON_CP_``,
    e.g. ``NANOTRON_CP_DATA_PLANE_URL=http://localhost:8080``.
    """

    model_config = SettingsConfigDict(
        env_prefix="NANOTRON_CP_",
        env_file=".env",
        extra="ignore",
    )

    bind_host: str = "0.0.0.0"
    bind_port: int = 8090

    data_plane_url: str = "http://localhost:8080"
    data_plane_timeout_s: float = 2.0

    config_root: Path = Field(default_factory=_default_config_root)
    snapshots_root: Path = Field(default=Path("artifacts/snapshots"))

    cors_allow_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    log_level: str = "INFO"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached settings accessor used as a FastAPI dependency."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

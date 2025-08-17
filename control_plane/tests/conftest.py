"""Test fixtures.

Each test gets its own isolated app + store rooted in a tempdir, so
parallel runs don't stomp on each other and there's no global state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nanotron_control.app import create_app
from nanotron_control.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "strategy.toml").write_text(
        """
[strategies.alpha]
enabled = true
risk_aversion = 0.4
max_position_usd = 500000
universe = ["AAPL", "MSFT"]
"""
    )
    (cfg / "risk.toml").write_text(
        """
[limits]
max_order_notional_usd = 750000
max_order_size = 50000
max_price_deviation_pct = 3.0
max_daily_loss_usd = 100000
kill_switch_enabled = true
"""
    )
    return Settings(
        config_root=cfg,
        snapshots_root=tmp_path / "snapshots",
        data_plane_url="http://127.0.0.1:1",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as c:
        yield c

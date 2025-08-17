"""Tests for the strategy lifecycle endpoints + state machine."""

from __future__ import annotations


def test_lists_strategies_loaded_from_toml(client):
    r = client.get("/strategies")
    assert r.status_code == 200
    names = [s["name"] for s in r.json()]
    assert "alpha" in names


def test_get_unknown_strategy_returns_404(client):
    r = client.get("/strategies/does-not-exist")
    assert r.status_code == 404


def test_upsert_rejects_path_body_mismatch(client):
    body = {"name": "beta", "enabled": True}
    r = client.put("/strategies/alpha", json=body)
    assert r.status_code == 400


def test_upsert_creates_then_updates(client):
    body = {"name": "beta", "enabled": True, "risk_aversion": 0.7}
    r = client.put("/strategies/beta", json=body)
    assert r.status_code == 200
    assert r.json()["risk_aversion"] == 0.7

    r2 = client.get("/strategies/beta")
    assert r2.status_code == 200
    assert r2.json()["risk_aversion"] == 0.7


def test_state_machine_legal_path(client):
    r = client.post("/strategies/alpha/transition", json={"target": "start"})
    assert r.status_code == 200
    assert r.json()["state"] == "running"

    r = client.post("/strategies/alpha/transition", json={"target": "pause"})
    assert r.status_code == 200
    assert r.json()["state"] == "paused"

    r = client.post("/strategies/alpha/transition", json={"target": "resume"})
    assert r.status_code == 200
    assert r.json()["state"] == "running"

    r = client.post("/strategies/alpha/transition", json={"target": "stop"})
    assert r.status_code == 200
    assert r.json()["state"] == "stopped"


def test_state_machine_rejects_illegal_transition(client):
    r = client.post("/strategies/alpha/transition", json={"target": "pause"})
    assert r.status_code == 409
    assert "illegal" in r.json()["detail"]


def test_state_machine_unknown_strategy_404(client):
    r = client.post(
        "/strategies/unknown/transition", json={"target": "start"}
    )
    assert r.status_code == 404


def test_universe_uppercased(client):
    body = {
        "name": "gamma",
        "enabled": True,
        "universe": ["aapl", "  msft ", "nvda"],
    }
    r = client.put("/strategies/gamma", json=body)
    assert r.status_code == 200
    assert r.json()["universe"] == ["AAPL", "MSFT", "NVDA"]


def test_strategy_name_validation(client):
    body = {"name": "bad name!", "enabled": True}
    r = client.put("/strategies/bad-name", json=body)
    assert r.status_code in (400, 422)

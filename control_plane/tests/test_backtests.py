"""Tests for backtest dispatch + status endpoints."""

from __future__ import annotations


def test_submit_returns_202_and_run_id(client):
    body = {
        "strategy": "alpha",
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-02-01T00:00:00Z",
        "universe": ["AAPL"],
        "capital_usd": 250000,
    }
    r = client.post("/backtests", json=body)
    assert r.status_code == 202
    payload = r.json()
    assert payload["status"] == "queued"
    assert len(payload["run_id"]) == 12


def test_submit_404_when_strategy_unknown(client):
    body = {
        "strategy": "ghost",
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-02-01T00:00:00Z",
        "capital_usd": 100,
    }
    r = client.post("/backtests", json=body)
    assert r.status_code == 404


def test_end_must_be_after_start(client):
    body = {
        "strategy": "alpha",
        "start": "2025-02-01T00:00:00Z",
        "end": "2025-01-01T00:00:00Z",
        "capital_usd": 100,
    }
    r = client.post("/backtests", json=body)
    assert r.status_code == 422


def test_listing_runs_orders_newest_first(client):
    for _ in range(3):
        client.post(
            "/backtests",
            json={
                "strategy": "alpha",
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-02-01T00:00:00Z",
                "capital_usd": 100,
            },
        )
    r = client.get("/backtests?limit=10")
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) == 3
    submitted = [run["submitted_at"] for run in runs]
    assert submitted == sorted(submitted, reverse=True)


def test_get_run_404_for_unknown(client):
    r = client.get("/backtests/000000000000")
    assert r.status_code == 404

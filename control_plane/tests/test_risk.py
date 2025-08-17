"""Tests for the /risk/limits resource."""

from __future__ import annotations


def test_limits_loaded_from_toml(client):
    r = client.get("/risk/limits")
    assert r.status_code == 200
    body = r.json()
    assert body["max_order_notional_usd"] == 750000
    assert body["max_order_size"] == 50000
    assert body["kill_switch_enabled"] is True


def test_limits_update_round_trip(client):
    new = {
        "max_order_notional_usd": 1_500_000,
        "max_order_size": 200_000,
        "max_price_deviation_pct": 7.5,
        "max_daily_loss_usd": 500_000,
        "kill_switch_enabled": False,
    }
    r = client.put("/risk/limits", json=new)
    assert r.status_code == 200
    body = r.json()
    assert body["max_order_notional_usd"] == 1_500_000
    assert body["kill_switch_enabled"] is False

    r2 = client.get("/risk/limits")
    assert r2.json()["max_daily_loss_usd"] == 500_000


def test_limits_reject_negative(client):
    r = client.put(
        "/risk/limits",
        json={
            "max_order_notional_usd": -1,
            "max_order_size": 10,
            "max_price_deviation_pct": 5,
            "max_daily_loss_usd": 0,
            "kill_switch_enabled": True,
        },
    )
    assert r.status_code == 422


def test_limits_reject_pct_over_100(client):
    r = client.put(
        "/risk/limits",
        json={
            "max_order_notional_usd": 1,
            "max_order_size": 1,
            "max_price_deviation_pct": 101,
            "max_daily_loss_usd": 0,
            "kill_switch_enabled": True,
        },
    )
    assert r.status_code == 422

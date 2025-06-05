import datetime as dt

import pytest

pytest.importorskip("respx")

import respx
from httpx import Response

from nanotron_data.connectors.polygon import PolygonClient


@pytest.mark.asyncio
async def test_polygon_returns_normalized_frame():
    client = PolygonClient(api_key="test")
    payload = {
        "results": [
            {"o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 1000, "n": 50, "vw": 100.2,
             "t": 1700000000000},
            {"o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 1500, "n": 60, "vw": 101.0,
             "t": 1700000060000},
        ]
    }
    with respx.mock(base_url="https://api.polygon.io") as mock:
        mock.get(url__regex=r"/v2/aggs/.*").respond(200, json=payload)
        df = await client.bars(
            "AAPL",
            dt.datetime(2023, 11, 14, tzinfo=dt.timezone.utc),
            dt.datetime(2023, 11, 15, tzinfo=dt.timezone.utc),
            frequency="1m",
        )
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "trades", "vwap"]
    assert len(df) == 2
    assert df.iloc[0]["close"] == 100.5


@pytest.mark.asyncio
async def test_polygon_5xx_triggers_retries_then_succeeds():
    client = PolygonClient(api_key="test")
    payload = {"results": []}
    with respx.mock(base_url="https://api.polygon.io") as mock:
        route = mock.get(url__regex=r"/v2/aggs/.*")
        route.side_effect = [
            Response(503, json={"error": "down"}),
            Response(503, json={"error": "down"}),
            Response(200, json=payload),
        ]
        df = await client.bars(
            "AAPL",
            dt.datetime(2023, 11, 14, tzinfo=dt.timezone.utc),
            dt.datetime(2023, 11, 15, tzinfo=dt.timezone.utc),
        )
    assert df.empty
    assert route.call_count == 3

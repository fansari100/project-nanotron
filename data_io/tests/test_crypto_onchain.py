import datetime as dt

import pytest

pytest.importorskip("respx")

import respx
from httpx import Response

from nanotron_data.connectors.binance import BinanceClient
from nanotron_data.connectors.hyperliquid import HyperliquidClient
from nanotron_data.connectors.onchain.web3_client import OnchainClient
from nanotron_data.connectors.onchain.uniswap_v3 import UniswapV3Pool, _twos_complement


@pytest.mark.asyncio
async def test_binance_normalizes_klines():
    client = BinanceClient()
    payload = [
        [1700000000000, "100.0", "101.0", "99.5", "100.5", "10.0",
         1700000059999, "1005.0", 50, "5.0", "502.5", "0"],
    ]
    with respx.mock(base_url="https://api.binance.com") as mock:
        mock.get(url__regex=r"/api/v3/klines.*").respond(200, json=payload)
        df = await client.bars(
            "BTCUSDT",
            dt.datetime(2023, 11, 14, tzinfo=dt.timezone.utc),
            dt.datetime(2023, 11, 15, tzinfo=dt.timezone.utc),
        )
    assert len(df) == 1
    row = df.iloc[0]
    assert row["close"] == 100.5
    assert row["trades"] == 50
    # vwap = quote_volume / volume = 1005 / 10 = 100.5
    assert row["vwap"] == 100.5


@pytest.mark.asyncio
async def test_hyperliquid_funding_history():
    client = HyperliquidClient()
    payload = [
        {"time": 1700000000000, "fundingRate": "0.0001", "premium": "0.00001",
         "oraclePx": "1500.0"},
    ]
    with respx.mock(base_url="https://api.hyperliquid.xyz") as mock:
        mock.post("/info").respond(200, json=payload)
        df = await client.funding_history(
            "ETH",
            dt.datetime(2023, 11, 14, tzinfo=dt.timezone.utc),
            dt.datetime(2023, 11, 15, tzinfo=dt.timezone.utc),
        )
    assert len(df) == 1
    assert df.iloc[0]["fundingRate"] == 0.0001


@pytest.mark.asyncio
async def test_onchain_block_number_and_balance():
    client = OnchainClient(rpc_url="https://mainnet.example/rpc")
    with respx.mock(base_url="https://mainnet.example") as mock:
        route = mock.post("/rpc")
        route.side_effect = [
            Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0x1234"}),
            Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0xde0b6b3a7640000"}),
        ]
        block = await client.block_number()
        bal = await client.balance("0xabc", "latest")
    assert block == 0x1234
    assert bal == 10**18


def test_twos_complement_signed_decode():
    assert _twos_complement(0x000001, 24) == 1
    assert _twos_complement(0xFFFFFF, 24) == -1
    assert _twos_complement(0x800000, 24) == -(1 << 23)

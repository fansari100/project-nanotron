"""Uniswap V3 pool helpers: read slot0 + tick range from a pool address.

We only need a couple of read paths to power the on-chain alpha
pipeline (mid-price, liquidity-weighted depth, recent swap volume).
For richer queries — including positions, fee tiers, and TWAP oracle
observations — escalate to The Graph or a dedicated Subgraph.
"""

from __future__ import annotations

from dataclasses import dataclass

from .web3_client import OnchainClient


# slot0() selector
_SLOT0 = "0x3850c7bd"
# liquidity()
_LIQ = "0x1a686502"


@dataclass
class UniswapV3Pool:
    client: OnchainClient
    pool_address: str

    async def slot0(self) -> dict:
        """Return parsed slot0: sqrtPriceX96, tick, observationIndex,
        observationCardinality, observationCardinalityNext, feeProtocol, unlocked."""
        raw = await self.client.call(self.pool_address, _SLOT0)
        # 7 packed fields
        data = raw[2:]  # strip 0x
        words = [int(data[i : i + 64], 16) for i in range(0, len(data), 64)]
        return {
            "sqrtPriceX96": words[0],
            "tick": _twos_complement(words[1], 24),
            "observationIndex": words[2],
            "observationCardinality": words[3],
            "observationCardinalityNext": words[4],
            "feeProtocol": words[5],
            "unlocked": bool(words[6]),
        }

    async def liquidity(self) -> int:
        raw = await self.client.call(self.pool_address, _LIQ)
        return int(raw, 16)

    async def mid_price(self, decimals_token0: int, decimals_token1: int) -> float:
        """Convert sqrtPriceX96 to a human-readable mid price (token1/token0)."""
        s = (await self.slot0())["sqrtPriceX96"]
        ratio = (s / (1 << 96)) ** 2
        return ratio * (10 ** decimals_token0) / (10 ** decimals_token1)


def _twos_complement(value: int, bits: int) -> int:
    """Decode a `bits`-wide two's-complement integer."""
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value

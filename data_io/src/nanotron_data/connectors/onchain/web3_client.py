"""Async JSON-RPC client for EVM chains.

Just enough surface area to cover what the research pipeline needs:

- ``block_number()``      latest block height
- ``get_logs(filter_)``   eth_getLogs, parsed as raw dicts
- ``call(tx)``            eth_call against a contract method
- ``balance(addr, block)``  ETH balance at a specific block

We deliberately don't pull in web3.py here — it's synchronous and
brings in an enormous transitive dependency tree.  This client speaks
the JSON-RPC wire protocol directly through httpx.AsyncClient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from ..base import (
    CircuitBreaker,
    TransientError,
    retry_policy,
)


@dataclass
class OnchainClient:
    rpc_url: str  # e.g. an Alchemy/Infura/QuickNode endpoint
    timeout_s: float = 10.0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    async def _rpc(self, method: str, params: list[Any]) -> Any:
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

        async def _call():
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                r = await c.post(self.rpc_url, json=body)
                if 500 <= r.status_code < 600:
                    raise TransientError(f"rpc {r.status_code}")
                r.raise_for_status()
                payload = r.json()
                if "error" in payload:
                    raise TransientError(str(payload["error"]))
                return payload["result"]

        async with self.breaker.guard():
            async for attempt in retry_policy():
                with attempt:
                    return await _call()
        return None

    async def block_number(self) -> int:
        return int(await self._rpc("eth_blockNumber", []), 16)

    async def get_logs(
        self,
        from_block: int,
        to_block: int,
        address: str | None = None,
        topics: list[str | None] | None = None,
    ) -> list[dict]:
        f: dict[str, Any] = {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
        }
        if address:
            f["address"] = address
        if topics:
            f["topics"] = topics
        return await self._rpc("eth_getLogs", [f])

    async def call(
        self, to: str, data: str, block: str | int = "latest"
    ) -> str:
        block_tag = hex(block) if isinstance(block, int) else block
        return await self._rpc("eth_call", [{"to": to, "data": data}, block_tag])

    async def balance(self, address: str, block: str | int = "latest") -> int:
        block_tag = hex(block) if isinstance(block, int) else block
        return int(await self._rpc("eth_getBalance", [address, block_tag]), 16)

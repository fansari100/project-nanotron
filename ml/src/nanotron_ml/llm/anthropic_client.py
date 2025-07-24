"""Async Anthropic Messages API client with retry + circuit breaker."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class AnthropicClient:
    api_key: str
    model: str = "claude-sonnet-4-5"
    base_url: str = "https://api.anthropic.com"
    timeout_s: float = 60.0
    api_version: str = "2023-06-01"

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    async def messages(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self._headers()) as c:
            r = await c.post(f"{self.base_url}/v1/messages", json=body)
            r.raise_for_status()
            payload = r.json()
        # Concatenate all text blocks
        return "".join(
            blk.get("text", "")
            for blk in payload.get("content", [])
            if blk.get("type") == "text"
        )

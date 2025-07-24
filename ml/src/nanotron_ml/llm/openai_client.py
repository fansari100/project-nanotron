"""Async OpenAI Responses API client.

Uses the v1/chat/completions endpoint (the most stable shape across
recent model families).  For function-calling/tool-use workflows,
upgrade to /v1/responses and pass the appropriate ``tools`` array.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class OpenAIClient:
    api_key: str
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com"
    timeout_s: float = 60.0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}

    async def chat(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self._headers()) as c:
            r = await c.post(f"{self.base_url}/v1/chat/completions", json=body)
            r.raise_for_status()
            payload = r.json()
        return payload["choices"][0]["message"]["content"]

"""Client for a local OpenAI-compatible inference server (vLLM, Ollama, TGI).

We don't pin a specific local engine — the OpenAI-compat REST shape is
the lingua franca for self-hosted inference.  ``base_url`` defaults to
vLLM's localhost port.  Auth optional (off in homelabs, on in production).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class LocalQwenClient:
    model: str = "Qwen2.5-72B-Instruct"
    base_url: str = "http://localhost:8000"
    timeout_s: float = 120.0
    api_key: str | None = None  # optional bearer token (e.g. for vLLM auth)

    def _headers(self) -> dict:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

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

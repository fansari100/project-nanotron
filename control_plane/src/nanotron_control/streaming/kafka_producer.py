"""Async Kafka producer wrapper.

Uses aiokafka.  Soft-imports so the package installs without it; raises
a clear error if anyone tries to ``connect()`` without the dep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    from aiokafka import AIOKafkaProducer  # type: ignore[import-not-found]

    _HAS_AIOKAFKA = True
except ImportError:
    AIOKafkaProducer = None  # type: ignore[misc, assignment]
    _HAS_AIOKAFKA = False


@dataclass
class KafkaProducerClient:
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "nanotron-control-plane"
    acks: str = "all"
    compression_type: str = "lz4"
    _producer: Any = None

    async def connect(self) -> None:
        if not _HAS_AIOKAFKA:
            raise RuntimeError("aiokafka not installed — pip install aiokafka")
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
            acks=self.acks,
            compression_type=self.compression_type,
        )
        await self._producer.start()

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def send(
        self, topic: str, value: dict | bytes, key: str | None = None
    ) -> None:
        if self._producer is None:
            raise RuntimeError("producer not connected")
        payload = value if isinstance(value, (bytes, bytearray)) else json.dumps(value).encode()
        await self._producer.send_and_wait(
            topic, payload, key=key.encode() if key else None
        )

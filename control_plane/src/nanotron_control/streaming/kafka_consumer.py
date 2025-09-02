"""Async Kafka consumer wrapper."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

try:
    from aiokafka import AIOKafkaConsumer  # type: ignore[import-not-found]

    _HAS_AIOKAFKA = True
except ImportError:
    AIOKafkaConsumer = None  # type: ignore[misc, assignment]
    _HAS_AIOKAFKA = False


@dataclass
class KafkaConsumerClient:
    topics: tuple[str, ...]
    group_id: str
    bootstrap_servers: str = "localhost:9092"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    _consumer: Any = None

    async def connect(self) -> None:
        if not _HAS_AIOKAFKA:
            raise RuntimeError("aiokafka not installed — pip install aiokafka")
        self._consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.auto_offset_reset,
            enable_auto_commit=self.enable_auto_commit,
        )
        await self._consumer.start()

    async def close(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

    async def __aiter__(self) -> AsyncIterator[dict]:
        if self._consumer is None:
            raise RuntimeError("consumer not connected")
        async for msg in self._consumer:
            try:
                value = json.loads(msg.value.decode())
            except Exception:
                value = msg.value
            yield {
                "topic": msg.topic,
                "partition": msg.partition,
                "offset": msg.offset,
                "key": msg.key.decode() if msg.key else None,
                "value": value,
            }
            if not self.enable_auto_commit:
                await self._consumer.commit()

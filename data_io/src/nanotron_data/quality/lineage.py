"""OpenLineage emitter — records "this dataset came from these inputs".

We don't take a hard dep on the openlineage-python client because most
research workflows just need the JSON envelope.  The emitter writes
events to either a POST endpoint or a local NDJSON log; both are
accepted by Marquez and any other OL backend.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx


@dataclass
class OpenLineageEvent:
    eventType: str  # START | COMPLETE | ABORT | FAIL
    job_namespace: str
    job_name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    facets: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    eventTime: float = field(default_factory=time.time)

    def to_envelope(self, producer: str) -> dict:
        return {
            "eventType": self.eventType,
            "eventTime": _iso(self.eventTime),
            "producer": producer,
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
            "run": {"runId": self.run_id, "facets": {}},
            "job": {
                "namespace": self.job_namespace,
                "name": self.job_name,
                "facets": self.facets,
            },
            "inputs": [{"namespace": "default", "name": n} for n in self.inputs],
            "outputs": [{"namespace": "default", "name": n} for n in self.outputs],
        }


def _iso(t: float) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).isoformat()


@dataclass
class LineageEmitter:
    """Either ``api_url`` or ``log_path`` (or both) must be provided."""

    api_url: str | None = None
    log_path: str | Path | None = None
    producer: str = "https://github.com/fansari100/project-nanotron"
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.api_url and not self.log_path:
            raise ValueError("LineageEmitter needs api_url and/or log_path")
        if self.log_path:
            self.log_path = Path(self.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: OpenLineageEvent) -> None:
        payload = event.to_envelope(self.producer)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        if self.api_url:
            with httpx.Client(timeout=self.timeout_s) as c:
                c.post(self.api_url, json=payload)

    def emit_complete(
        self,
        job_namespace: str,
        job_name: str,
        inputs: list[str],
        outputs: list[str],
        facets: dict | None = None,
    ) -> str:
        ev = OpenLineageEvent(
            eventType="COMPLETE",
            job_namespace=job_namespace,
            job_name=job_name,
            inputs=inputs,
            outputs=outputs,
            facets=facets or {},
        )
        self.emit(ev)
        return ev.run_id

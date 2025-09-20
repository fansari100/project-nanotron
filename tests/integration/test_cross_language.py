"""Cross-language integration test for the ring-buffer ABI.

Pins down the contract between the Python producer (this file) and the
Rust consumer (`execution/src/shared_memory.rs`).  Skipped automatically
if cargo isn't on PATH so the suite still runs in environments without
a Rust toolchain.

The test writes N signals into a temp file using the canonical Python
producer, then shells out to a tiny Rust binary that mmaps the same
file and prints what it reads.  We compare line-for-line.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .ring_writer import RingWriter, TradingSignal

CARGO = shutil.which("cargo")
RUST_HARNESS_DIR = Path(__file__).parent / "rust_harness"

needs_cargo = pytest.mark.skipif(CARGO is None, reason="cargo not available")


@pytest.fixture(scope="module")
def harness_binary(tmp_path_factory) -> Path:
    """Build a tiny rust binary that re-uses execution/src/shared_memory.rs."""
    target_dir = tmp_path_factory.mktemp("rust_target")
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    out = subprocess.run(
        [CARGO, "build", "--release", "--manifest-path", str(RUST_HARNESS_DIR / "Cargo.toml")],
        env=env,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        pytest.skip(f"rust harness build failed: {out.stderr[-2000:]}")
    return target_dir / "release" / "ring-harness"


def _run(harness: Path, ring_path: Path) -> list[dict]:
    out = subprocess.run(
        [str(harness), str(ring_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if out.returncode != 0:
        raise RuntimeError(f"harness exited {out.returncode}: {out.stderr}")
    parsed = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        parts = dict(p.split("=") for p in line.split(","))
        parsed.append(parts)
    return parsed


@needs_cargo
def test_python_writer_to_rust_reader(tmp_path: Path, harness_binary: Path) -> None:
    ring = tmp_path / "nanotron_ring"
    writer = RingWriter(ring, max_records=64)
    writer.init()

    signals = [
        TradingSignal(ticker_id=i, direction=(i % 3) - 1, confidence=0.5 + 0.01 * i,
                      size=100.0 * (i + 1), reasoning_depth=i, latency_us=i * 10)
        for i in range(8)
    ]
    for s in signals:
        writer.append(s)

    read = _run(harness_binary, ring)
    assert len(read) == len(signals), f"expected {len(signals)}, got {len(read)}"

    for src, dst in zip(signals, read):
        assert int(dst["ticker_id"]) == src.ticker_id
        assert int(dst["direction"]) == src.direction
        assert int(dst["reasoning"]) == src.reasoning_depth
        assert int(dst["latency_us"]) == src.latency_us
        # Confidence/size are f32 — bit-exact roundtrip not guaranteed
        # through text formatting; tolerate small epsilon.
        assert abs(float(dst["confidence"]) - src.confidence) < 1e-5
        assert abs(float(dst["size"]) - src.size) < 1e-3


def test_python_writer_layout_is_stable() -> None:
    """If this asserts breaks, the on-wire format changed — bump the
    ABI version with the rust producer in lockstep before merging."""
    s = TradingSignal(
        ticker_id=0xCAFEBABE & 0xFFFFFFFF,
        direction=1,
        confidence=1.0,
        size=2.0,
        reasoning_depth=3,
        latency_us=4,
    )
    blob = s.to_bytes()
    assert len(blob) == 32
    assert blob[:4] == b"\xbe\xba\xfe\xca"
    assert blob[4] == 0x01

"""Reference implementation of the producer side of the ring-buffer ABI.

This is the Python mirror of what the Mojo/C++ producer does on a real
deployment.  It exists so that the rust-side reader has something to
talk to in CI without bringing up the full strategy core, and so that
this file is the single, executable spec the cross-language integration
test pins against.

Header layout (little-endian, repr(C)):

    offset  size  field
    ------  ----  ---------
       0     8    magic        = 0x4E414E4F54524F4E   ("NANOTRON")
       8     8    write_pos    monotonically incrementing
      16     8    read_pos     reserved (consumer-side cursor)
      24     8    record_size  = 32
      32     8    max_records  ring depth
      40     P    body         max_records * record_size

Record layout (32 bytes, little-endian):

    offset  size  field
    ------  ----  -----------
       0     4    ticker_id     u32
       4     1    direction     i8 (-1, 0, 1)
       5     3    -- padding
       8     4    confidence    f32
      12     4    size          f32
      16     4    reasoning     i32
      20     8    latency_us    i64
      28     4    -- padding (timestamp_ns is set on the consumer side)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = 0x4E414E4F54524F4E
HEADER_FMT = "<QQQQQ"  # 5 x u64 little-endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_SIZE = 32
RECORD_FMT = "<IbxxxffiqI"  # u32, i8, 3xpad, f32, f32, i32, i64, u32(pad)
assert HEADER_SIZE == 40
assert struct.calcsize(RECORD_FMT) == RECORD_SIZE


@dataclass
class TradingSignal:
    ticker_id: int
    direction: int
    confidence: float
    size: float
    reasoning_depth: int
    latency_us: int

    def to_bytes(self) -> bytes:
        return struct.pack(
            RECORD_FMT,
            self.ticker_id,
            self.direction,
            self.confidence,
            self.size,
            self.reasoning_depth,
            self.latency_us,
            0,
        )


class RingWriter:
    """Build / append-to / inspect a NANOTRON ring buffer file."""

    def __init__(self, path: Path, max_records: int = 1024) -> None:
        self.path = Path(path)
        self.max_records = max_records
        self.write_pos = 0

    def init(self) -> None:
        size = HEADER_SIZE + self.max_records * RECORD_SIZE
        with self.path.open("wb") as f:
            f.write(self._header(write_pos=0))
            f.write(b"\x00" * (size - HEADER_SIZE))

    def append(self, signal: TradingSignal) -> None:
        with self.path.open("r+b") as f:
            slot = self.write_pos % self.max_records
            f.seek(HEADER_SIZE + slot * RECORD_SIZE)
            f.write(signal.to_bytes())
            self.write_pos += 1
            f.seek(0)
            f.write(self._header(write_pos=self.write_pos))

    def _header(self, write_pos: int) -> bytes:
        return struct.pack(
            HEADER_FMT, MAGIC, write_pos, 0, RECORD_SIZE, self.max_records
        )

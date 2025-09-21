# ADR-0001: Typed C-ABI signal contract over `/dev/shm` ring buffer

- Status: accepted
- Date: 2025-08-03

## Context

The strategy core (Mojo + JAX) and the consumer (Rust + Axum) live in
different processes and different languages. They have to exchange
millions of decisions a day with sub-microsecond per-record overhead.
Three options were on the table:

1. **TCP/UDP socket** between the two processes — simple but each record
   pays a syscall, and we lose the page cache.
2. **Apache Arrow Flight stream** — first-class typed columnar streams,
   but Flight is throughput-oriented and adds RPC framing per batch.
3. **Memory-mapped ring buffer in `/dev/shm`** with a fixed binary
   record layout — the producer writes 32 bytes, the consumer mmaps
   the file read-only and walks the cursor. No syscalls per record,
   no copies.

## Decision

Option 3, with two additional constraints:

- A `#[repr(C)]` header (40 bytes) prefixes the body. Magic number
  `0x4E414E4F54524F4E` (ASCII "NANOTRON"), monotonically incrementing
  `write_pos`, reserved `read_pos`, fixed `record_size = 32`,
  `max_records` set at producer init.
- Records are little-endian and field-aligned by hand. The single
  source of truth for the layout is `tests/integration/ring_writer.py`,
  which is exercised by a cross-language test in CI.

## Consequences

- The hot path is one mmap'd read per record. Criterion shows
  `from_bytes` at <50 ns and a full ring-buffer read at <120 ns.
- We accept that adding or reordering a field is a breaking ABI change.
  The proptest suite and the layout-stability assertion in
  `test_python_writer_layout_is_stable` are the guard rails.
- Backpressure is the producer's job — when the consumer falls behind,
  the producer overwrites old slots. Acceptable because every record is
  also broadcast over the websocket; downstream consumers that need
  history go through the control plane / kdb+, not the ring.

# ADR-0002: Rust + Axum for the data-plane middleware

- Status: accepted
- Date: 2025-08-04

## Context

The data plane sits between two failure-mode regimes:

- **Below it**, the producer is a hot loop in C++/Mojo. Memory layouts
  matter; a partial read or a misaligned field is silent corruption.
- **Above it**, browser clients connect over WebSocket. A slow client
  must not back-pressure the producer; transient network errors must
  not crash the process.

The middleware needs to enforce the boundary cleanly: deserialize the
producer's bytes into typed values, fan them out over WebSocket with
bounded backpressure, expose health/readiness/metrics for k8s.

## Decision

Rust, with `axum` (built on `tower` + `hyper`) for the HTTP/WS surface.

Specific choices that fell out of this:

- `tokio::sync::broadcast` for the fan-out, with explicit drop-laggy
  semantics. A websocket client that lags more than the channel
  capacity gets `RecvError::Lagged` and we drop the connection rather
  than letting it back-pressure the producer.
- `memmap2::Mmap` (read-only) over the ring buffer. No `unsafe` outside
  the mmap call, no allocations on the hot path.
- `prometheus`-format `/metrics` rendered by hand into a `String`
  rather than pulling in the full `prometheus` crate's macro machinery
  for ~10 metrics.
- Graceful shutdown on `SIGTERM` and `ctrl_c` via `tokio::select!` so
  k8s rolls the deployment cleanly.

## Alternatives considered

- **Python (FastAPI/uvicorn)**: lifetime/aliasing rules around mmap'd
  bytes are exactly what Rust's borrow checker is for. Doing this in
  Python means either copies on every read (kills latency) or `ctypes`
  + raw pointers (gives up the safety benefit anyway).
- **Go**: viable. Lost out because Mojo/JAX/Python are already in the
  process tree; adding Go would mean three runtimes. Rust pairs more
  naturally with the existing C++ producer.
- **Node + ws**: V8's GC is in the wrong place for this workload.

## Consequences

- Reviewers without Rust experience need a tour of `tokio::broadcast`
  semantics and `axum`'s `State<>` extractor. Mitigated by keeping
  `main.rs` short — every handler is <20 lines and fits on a screen.
- Rust toolchain in CI adds ~2 minutes of wall-clock per run. Cached
  via `Swatinem/rust-cache` so steady-state is <30s.

# Architecture Decision Records

These are short, dated, append-only notes about the architectural
choices that are most likely to come up in code review or interviews.
Each one captures *the alternative we picked against* and *why*, so we
don't relitigate the decision every quarter.

| ADR | Title |
|-----|-------|
| [0001](./0001-typed-c-abi-signal-contract.md) | Typed C-ABI signal contract over `/dev/shm` ring buffer |
| [0002](./0002-rust-middleware-axum.md) | Rust + Axum for the data-plane middleware |
| [0003](./0003-control-plane-vs-data-plane-split.md) | Control plane (FastAPI) split from data plane (Rust) |
| [0004](./0004-formal-verification-cpp-hot-path.md) | Frama-C / ACSL on the C++ order gateway |

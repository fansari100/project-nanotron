# nanotron control plane

Non-real-time HTTP surface for project-nanotron. Owns:

- `/strategies` — list/upsert + a typed state machine (idle → running → paused → stopped)
- `/risk/limits` — read/update the global risk envelope (mirrors `config/risk.toml`)
- `/backtests` — submit + list + inspect backtest runs (queued in-process for now)
- `/snapshots` — read-only snapshot inventory
- `/health`, `/ready`, `/status` — fan out to the Rust data plane (`http://localhost:8080` by default)
- `/metrics` — Prometheus exposition

The Rust data plane (`execution/`) is the hot path; this service intentionally
sits *off* it so changes to operator-facing semantics (auth, validation, audit)
don't require touching the Rust binary.

## Run

```bash
pip install -e .[dev]
python -m nanotron_control                    # bind 0.0.0.0:8090
open http://localhost:8090/docs               # OpenAPI swagger
```

Configuration is via env vars prefixed with `NANOTRON_CP_`, or a `.env` file:

| Var | Default | Notes |
|---|---|---|
| `NANOTRON_CP_BIND_HOST` | `0.0.0.0` | |
| `NANOTRON_CP_BIND_PORT` | `8090` | |
| `NANOTRON_CP_DATA_PLANE_URL` | `http://localhost:8080` | Rust axum service |
| `NANOTRON_CP_DATA_PLANE_TIMEOUT_S` | `2.0` | |
| `NANOTRON_CP_CONFIG_ROOT` | repo `config/` | Source of truth for TOMLs |
| `NANOTRON_CP_LOG_LEVEL` | `INFO` | |

## Test

```bash
pytest -q
```

State is per-process and resets on restart; the on-disk TOMLs are the durable
source of truth and are loaded at boot.

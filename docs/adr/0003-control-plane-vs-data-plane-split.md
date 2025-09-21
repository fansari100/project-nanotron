# ADR-0003: Control plane (FastAPI) split from data plane (Rust)

- Status: accepted
- Date: 2025-08-15

## Context

The original v0.1.0 had every HTTP endpoint on the Rust binary. That
worked for `/health` and `/status`, but as soon as we needed:

- a strategy state machine with audit trails,
- editable risk limits with validation messages,
- backtest dispatch + history,
- snapshot inventory,

…the cost/benefit shifted. These are operator-facing endpoints that
iterate fast (auth rules, validation messages, audit hooks) and don't
need to be on the websocket-streaming hot path.

## Decision

Split the surface in two:

- **Data plane** (Rust, `:8080`) — websocket fan-out from shared
  memory, plus the four endpoints k8s needs: `/health`, `/ready`,
  `/status`, `/metrics`. Nothing else lives here.
- **Control plane** (Python + FastAPI, `:8090`) — strategy lifecycle,
  risk limits, backtests, snapshots. Talks to the data plane only
  through its public HTTP surface.

The frontend talks to both: WebSocket → data plane, REST → control
plane. The Helm chart's ingress splits by path:

| path | service |
|---|---|
| `/ws` | data-plane |
| `/api/control` | control-plane |
| `/` | frontend |

## Consequences

- Two services to operate. The Helm chart and `docker-compose.yml`
  bring them up together so this is a non-cost in dev/staging.
- The contract between the two is small (the data plane's public REST
  surface) and pinned in the FastAPI app's `DataPlaneClient`.
- Operator-facing endpoints can ship in a release without rebuilding
  the Rust binary. This was the dominant motivation.
- Adding Postgres/Redis on the control plane is a localized change;
  it can never affect the hot path.

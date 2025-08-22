-- Initial schema for the control plane's persistent state.
-- TimescaleDB extension is optional but enabled when present.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS strategies (
    name             TEXT PRIMARY KEY,
    enabled          BOOLEAN NOT NULL DEFAULT TRUE,
    state            TEXT NOT NULL DEFAULT 'idle',
    risk_aversion    DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    max_position_usd DOUBLE PRECISION NOT NULL DEFAULT 1000000.0,
    universe         TEXT[] NOT NULL DEFAULT '{}',
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_limits (
    id                       INTEGER PRIMARY KEY,  -- always 1
    max_order_notional_usd   DOUBLE PRECISION NOT NULL,
    max_order_size           INTEGER NOT NULL,
    max_price_deviation_pct  DOUBLE PRECISION NOT NULL,
    max_daily_loss_usd       DOUBLE PRECISION NOT NULL,
    kill_switch_enabled      BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (id = 1)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id        TEXT PRIMARY KEY,
    strategy      TEXT NOT NULL REFERENCES strategies(name) ON DELETE CASCADE,
    status        TEXT NOT NULL,
    submitted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at  TIMESTAMPTZ,
    pnl_usd       DOUBLE PRECISION,
    sharpe        DOUBLE PRECISION,
    error         TEXT
);

-- Audit log — append-only, hypertable when timescale is present.
CREATE TABLE IF NOT EXISTS audit_log (
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    resource    TEXT NOT NULL,
    detail      JSONB
);
SELECT create_hypertable('audit_log', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- High-frequency signal table: one row per published signal.
CREATE TABLE IF NOT EXISTS signals (
    ts          TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    signal      DOUBLE PRECISION NOT NULL,
    confidence  DOUBLE PRECISION
);
SELECT create_hypertable('signals', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
CREATE INDEX IF NOT EXISTS signals_symbol_ts_idx ON signals (symbol, ts DESC);

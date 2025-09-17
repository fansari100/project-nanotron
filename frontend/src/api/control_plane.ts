/**
 * Typed client for the FastAPI control plane.
 *
 * Wraps fetch + zod for runtime parsing.  All methods are idempotent
 * over the network: the React-Query cache decides what to refetch and
 * when.  Errors throw an Error with a status-augmented message.
 */
import { z } from "zod";

const StrategyState = z.enum(["idle", "running", "paused", "stopped", "error"]);

export const Strategy = z.object({
  name: z.string(),
  enabled: z.boolean(),
  state: StrategyState,
  risk_aversion: z.number(),
  max_position_usd: z.number(),
  universe: z.array(z.string()),
  updated_at: z.string(),
});
export type Strategy = z.infer<typeof Strategy>;

export const RiskLimits = z.object({
  max_order_notional_usd: z.number(),
  max_order_size: z.number(),
  max_price_deviation_pct: z.number(),
  max_daily_loss_usd: z.number(),
  kill_switch_enabled: z.boolean(),
  updated_at: z.string(),
});
export type RiskLimits = z.infer<typeof RiskLimits>;

export const BacktestRunStatus = z.enum(["queued", "running", "complete", "failed"]);
export const BacktestRun = z.object({
  run_id: z.string(),
  strategy: z.string(),
  status: BacktestRunStatus,
  submitted_at: z.string(),
  completed_at: z.string().nullable().optional(),
  pnl_usd: z.number().nullable().optional(),
  sharpe: z.number().nullable().optional(),
  error: z.string().nullable().optional(),
});
export type BacktestRun = z.infer<typeof BacktestRun>;

const HealthReport = z.object({
  status: z.enum(["healthy", "degraded"]),
  control_plane: z.record(z.unknown()),
  data_plane: z.record(z.unknown()),
});
export type HealthReport = z.infer<typeof HealthReport>;

export class ControlPlaneClient {
  constructor(
    private readonly baseUrl: string,
    private readonly token?: string,
  ) {}

  private async req<T>(path: string, init: RequestInit, schema: z.ZodSchema<T>): Promise<T> {
    const headers = new Headers(init.headers);
    if (this.token) headers.set("Authorization", `Bearer ${this.token}`);
    headers.set("Accept", "application/json");
    if (init.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    const r = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    if (!r.ok) {
      const body = await r.text();
      throw new Error(`HTTP ${r.status} on ${path}: ${body.slice(0, 256)}`);
    }
    return schema.parse(await r.json());
  }

  health = (): Promise<HealthReport> =>
    this.req("/health", { method: "GET" }, HealthReport);

  listStrategies = (): Promise<Strategy[]> =>
    this.req("/strategies", { method: "GET" }, z.array(Strategy));

  getStrategy = (name: string): Promise<Strategy> =>
    this.req(`/strategies/${encodeURIComponent(name)}`, { method: "GET" }, Strategy);

  upsertStrategy = (s: Strategy): Promise<Strategy> =>
    this.req(
      `/strategies/${encodeURIComponent(s.name)}`,
      { method: "PUT", body: JSON.stringify(s) },
      Strategy,
    );

  transition = (
    name: string,
    target: "start" | "pause" | "resume" | "stop",
    reason?: string,
  ): Promise<Strategy> =>
    this.req(
      `/strategies/${encodeURIComponent(name)}/transition`,
      { method: "POST", body: JSON.stringify({ target, reason }) },
      Strategy,
    );

  getRiskLimits = (): Promise<RiskLimits> =>
    this.req("/risk/limits", { method: "GET" }, RiskLimits);

  updateRiskLimits = (r: RiskLimits): Promise<RiskLimits> =>
    this.req("/risk/limits", { method: "PUT", body: JSON.stringify(r) }, RiskLimits);

  listBacktests = (limit = 50): Promise<BacktestRun[]> =>
    this.req(`/backtests?limit=${limit}`, { method: "GET" }, z.array(BacktestRun));

  submitBacktest = (body: {
    strategy: string;
    start: string;
    end: string;
    universe?: string[];
    capital_usd?: number;
  }): Promise<BacktestRun> =>
    this.req("/backtests", { method: "POST", body: JSON.stringify(body) }, BacktestRun);
}

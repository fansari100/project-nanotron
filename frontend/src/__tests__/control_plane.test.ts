import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { ControlPlaneClient } from "../api/control_plane";

describe("ControlPlaneClient", () => {
  const fetchMock = vi.fn();
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    fetchMock.mockReset();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("listStrategies parses a valid response", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify([
          {
            name: "alpha",
            enabled: true,
            state: "running",
            risk_aversion: 0.4,
            max_position_usd: 1000000,
            universe: ["AAPL"],
            updated_at: "2025-09-10T10:00:00Z",
          },
        ]),
        { status: 200 },
      ),
    );
    const c = new ControlPlaneClient("http://x");
    const out = await c.listStrategies();
    expect(out).toHaveLength(1);
    expect(out[0].state).toBe("running");
  });

  it("throws on 4xx", async () => {
    fetchMock.mockResolvedValueOnce(new Response("boom", { status: 500 }));
    const c = new ControlPlaneClient("http://x");
    await expect(c.listStrategies()).rejects.toThrow(/HTTP 500/);
  });

  it("rejects schema-invalid responses", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify([{ name: "alpha" /* missing required fields */ }]),
        { status: 200 },
      ),
    );
    const c = new ControlPlaneClient("http://x");
    await expect(c.listStrategies()).rejects.toThrow();
  });

  it("transition POSTs the right body", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          name: "alpha",
          enabled: true,
          state: "running",
          risk_aversion: 0.5,
          max_position_usd: 1,
          universe: [],
          updated_at: "2025-09-10T10:00:00Z",
        }),
        { status: 200 },
      ),
    );
    const c = new ControlPlaneClient("http://x");
    await c.transition("alpha", "start", "manual ops");
    const lastCall = fetchMock.mock.calls[0];
    expect(lastCall[0]).toBe("http://x/strategies/alpha/transition");
    expect(lastCall[1]?.method).toBe("POST");
    expect(JSON.parse(lastCall[1]?.body as string)).toEqual({
      target: "start",
      reason: "manual ops",
    });
  });

  it("attaches bearer token when provided", async () => {
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          status: "healthy",
          control_plane: { ok: true },
          data_plane: { ok: true },
        }),
        { status: 200 },
      ),
    );
    const c = new ControlPlaneClient("http://x", "tok123");
    await c.health();
    const headers = fetchMock.mock.calls[0][1]?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer tok123");
  });
});

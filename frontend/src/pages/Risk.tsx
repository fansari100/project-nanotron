/** Risk-limits editor. */
import { useState, useEffect } from "react";

import { ControlPlaneClient, type RiskLimits } from "../api/control_plane";

const client = new ControlPlaneClient(
  import.meta.env.VITE_CONTROL_PLANE_URL ?? "http://localhost:8090",
);

export default function Risk() {
  const [limits, setLimits] = useState<RiskLimits | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    client
      .getRiskLimits()
      .then(setLimits)
      .catch((e) => setError((e as Error).message));
  }, []);

  if (error) return <div className="p-4 text-red-500">{error}</div>;
  if (!limits) return <div className="p-4">Loading…</div>;

  const fields: Array<[keyof RiskLimits, string, number, number]> = [
    ["max_order_notional_usd", "Max order notional ($)", 0, 1e9],
    ["max_order_size", "Max order size (shares)", 0, 1e8],
    ["max_price_deviation_pct", "Max price deviation (%)", 0, 100],
    ["max_daily_loss_usd", "Max daily loss ($)", 0, 1e9],
  ];

  return (
    <div className="p-4 max-w-xl">
      <h1 className="text-2xl font-bold mb-4">Risk limits</h1>
      <form
        className="space-y-3"
        onSubmit={async (e) => {
          e.preventDefault();
          const updated = await client.updateRiskLimits(limits);
          setLimits(updated);
        }}
      >
        {fields.map(([k, label, min, max]) => (
          <div key={k} className="flex justify-between">
            <label className="font-medium">{label}</label>
            <input
              type="number"
              min={min}
              max={max}
              value={Number(limits[k] as number)}
              className="border px-2 py-1 w-48 text-right"
              onChange={(e) =>
                setLimits({ ...limits, [k]: Number(e.target.value) })
              }
            />
          </div>
        ))}
        <div className="flex justify-between">
          <label className="font-medium">Kill-switch enabled</label>
          <input
            type="checkbox"
            checked={limits.kill_switch_enabled}
            onChange={(e) =>
              setLimits({ ...limits, kill_switch_enabled: e.target.checked })
            }
          />
        </div>
        <button
          type="submit"
          className="px-3 py-1 bg-blue-600 text-white rounded mt-3"
        >
          Save
        </button>
      </form>
    </div>
  );
}

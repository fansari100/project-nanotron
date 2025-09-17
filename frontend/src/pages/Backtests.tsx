/** Backtests page — submit a run and watch the queue. */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { ControlPlaneClient } from "../api/control_plane";

const client = new ControlPlaneClient(
  import.meta.env.VITE_CONTROL_PLANE_URL ?? "http://localhost:8090",
);

export default function Backtests() {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["backtests"],
    queryFn: () => client.listBacktests(50),
    refetchInterval: 3_000,
  });

  const submit = useMutation({
    mutationFn: client.submitBacktest,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["backtests"] }),
  });

  const [strategy, setStrategy] = useState("alpha");
  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState("2024-06-30");

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Backtests</h1>
      <form
        className="flex space-x-2 mb-4"
        onSubmit={(e) => {
          e.preventDefault();
          submit.mutate({
            strategy,
            start: `${start}T00:00:00Z`,
            end: `${end}T00:00:00Z`,
          });
        }}
      >
        <input
          className="border px-2 py-1"
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          placeholder="strategy"
        />
        <input
          className="border px-2 py-1"
          type="date"
          value={start}
          onChange={(e) => setStart(e.target.value)}
        />
        <input
          className="border px-2 py-1"
          type="date"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
        />
        <button className="px-3 py-1 bg-blue-600 text-white rounded">Run</button>
      </form>
      <table className="min-w-full border">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-2 py-1 text-left">Run</th>
            <th className="px-2 py-1 text-left">Strategy</th>
            <th className="px-2 py-1 text-left">Status</th>
            <th className="px-2 py-1 text-right">PnL ($)</th>
            <th className="px-2 py-1 text-right">Sharpe</th>
            <th className="px-2 py-1 text-left">Submitted</th>
          </tr>
        </thead>
        <tbody>
          {data?.map((r) => (
            <tr key={r.run_id} className="border-t">
              <td className="px-2 py-1 font-mono">{r.run_id}</td>
              <td className="px-2 py-1">{r.strategy}</td>
              <td className="px-2 py-1">{r.status}</td>
              <td className="px-2 py-1 text-right">
                {r.pnl_usd != null ? r.pnl_usd.toFixed(0) : "–"}
              </td>
              <td className="px-2 py-1 text-right">
                {r.sharpe != null ? r.sharpe.toFixed(2) : "–"}
              </td>
              <td className="px-2 py-1">{new Date(r.submitted_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

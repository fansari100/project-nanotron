/**
 * Strategies page — list view with state-machine controls.
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import { ControlPlaneClient } from "../api/control_plane";

const client = new ControlPlaneClient(
  import.meta.env.VITE_CONTROL_PLANE_URL ?? "http://localhost:8090",
);

export default function Strategies() {
  const qc = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ["strategies"],
    queryFn: () => client.listStrategies(),
    refetchInterval: 5_000,
  });

  const transition = useMutation({
    mutationFn: ({
      name,
      target,
    }: {
      name: string;
      target: "start" | "pause" | "resume" | "stop";
    }) => client.transition(name, target),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["strategies"] }),
  });

  if (isLoading) return <div className="p-4">Loading…</div>;
  if (error) return <div className="p-4 text-red-500">Failed to load strategies</div>;

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Strategies</h1>
      <table className="min-w-full border">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-2 py-1 text-left">Name</th>
            <th className="px-2 py-1 text-left">State</th>
            <th className="px-2 py-1 text-right">Risk γ</th>
            <th className="px-2 py-1 text-right">Max position $</th>
            <th className="px-2 py-1 text-left">Actions</th>
          </tr>
        </thead>
        <tbody>
          {data?.map((s) => (
            <tr key={s.name} className="border-t">
              <td className="px-2 py-1 font-mono">{s.name}</td>
              <td className="px-2 py-1">
                <span
                  className={`px-2 py-1 rounded text-xs ${
                    s.state === "running"
                      ? "bg-green-100"
                      : s.state === "paused"
                        ? "bg-yellow-100"
                        : "bg-gray-100"
                  }`}
                >
                  {s.state}
                </span>
              </td>
              <td className="px-2 py-1 text-right">{s.risk_aversion.toFixed(2)}</td>
              <td className="px-2 py-1 text-right">
                ${s.max_position_usd.toLocaleString()}
              </td>
              <td className="px-2 py-1 space-x-1">
                {s.state === "idle" || s.state === "stopped" ? (
                  <button
                    className="px-2 py-1 bg-blue-500 text-white rounded text-xs"
                    onClick={() =>
                      transition.mutate({ name: s.name, target: "start" })
                    }
                  >
                    Start
                  </button>
                ) : null}
                {s.state === "running" ? (
                  <>
                    <button
                      className="px-2 py-1 bg-yellow-500 text-white rounded text-xs"
                      onClick={() =>
                        transition.mutate({ name: s.name, target: "pause" })
                      }
                    >
                      Pause
                    </button>
                    <button
                      className="px-2 py-1 bg-red-500 text-white rounded text-xs"
                      onClick={() =>
                        transition.mutate({ name: s.name, target: "stop" })
                      }
                    >
                      Stop
                    </button>
                  </>
                ) : null}
                {s.state === "paused" ? (
                  <button
                    className="px-2 py-1 bg-green-500 text-white rounded text-xs"
                    onClick={() =>
                      transition.mutate({ name: s.name, target: "resume" })
                    }
                  >
                    Resume
                  </button>
                ) : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

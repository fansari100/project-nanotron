/** Models page — registry view (placeholder for the Dec 8 MLflow integration). */
import { useEffect, useState } from "react";

interface ModelEntry {
  name: string;
  version: string;
  stage: "Staging" | "Production" | "Archived";
  uri: string;
  rmse?: number;
  sharpe?: number;
}

const FALLBACK: ModelEntry[] = [
  { name: "signal-tft", version: "v3", stage: "Production", uri: "models:/signal-tft/3", rmse: 0.0042, sharpe: 1.81 },
  { name: "signal-tft", version: "v4", stage: "Staging",    uri: "models:/signal-tft/4", rmse: 0.0040, sharpe: 1.93 },
  { name: "regime-moe", version: "v2", stage: "Production", uri: "models:/regime-moe/2", sharpe: 1.42 },
];

export default function Models() {
  const [models, setModels] = useState<ModelEntry[] | null>(null);
  useEffect(() => {
    // The MLflow REST proxy lives behind /api/control/mlflow in prod; in dev
    // we ship sample data so the page always renders.
    setModels(FALLBACK);
  }, []);
  if (!models) return <div className="p-4">Loading…</div>;
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Models</h1>
      <table className="min-w-full border">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-2 py-1 text-left">Name</th>
            <th className="px-2 py-1 text-left">Version</th>
            <th className="px-2 py-1 text-left">Stage</th>
            <th className="px-2 py-1 text-right">RMSE</th>
            <th className="px-2 py-1 text-right">Sharpe</th>
            <th className="px-2 py-1 text-left">URI</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={`${m.name}/${m.version}`} className="border-t">
              <td className="px-2 py-1 font-mono">{m.name}</td>
              <td className="px-2 py-1">{m.version}</td>
              <td className="px-2 py-1">{m.stage}</td>
              <td className="px-2 py-1 text-right">{m.rmse?.toFixed(4) ?? "–"}</td>
              <td className="px-2 py-1 text-right">{m.sharpe?.toFixed(2) ?? "–"}</td>
              <td className="px-2 py-1 font-mono text-xs">{m.uri}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

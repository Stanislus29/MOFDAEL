import { useEffect, useRef, useState } from "react";
import { api } from "../api/client.js";
import { Button, Chip, Hairline, SectionTitle } from "../components/ui.jsx";
import { IconUpload } from "../components/icons.jsx";

export default function LabView() {
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [selected, setSelected] = useState([]);
  const [datasetId, setDatasetId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [runResult, setRunResult] = useState(null);
  const fileRef = useRef(null);

  useEffect(() => {
    api.listModels().then((r) => setModels(r.models)).catch(() => {});
    api.listDatasets().then((r) => {
      setDatasets(r.datasets);
      if (r.datasets.length) setDatasetId((id) => id || r.datasets[0].dataset_id);
    }).catch(() => {});
  }, []);

  const toggleModel = (id) =>
    setSelected((s) =>
      s.includes(id) ? s.filter((x) => x !== id) : [...s, id]
    );

  const selectAll = () => setSelected(models.filter((m) => m.available).map((m) => m.model_id));

  const upload = async (file) => {
    setBusy(true);
    setError("");
    try {
      const res = await api.uploadDataset(file);
      setDatasetId(res.dataset_id);
      const list = await api.listDatasets();
      setDatasets(list.datasets);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const run = async () => {
    setBusy(true);
    setError("");
    setRunResult(null);
    try {
      const res = await api.startRun({
        dataset_id: datasetId,
        model_ids: selected,
      });
      setRunResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const summary = runResult?.summary;

  return (
    <div>
      <SectionTitle>1 · Measurement data</SectionTitle>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <select
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
          style={selectStyle}
        >
          <option value="">Select a dataset…</option>
          {datasets.map((d) => (
            <option key={d.dataset_id} value={d.dataset_id}>
              {d.filename} · {d.n_rows} rows
            </option>
          ))}
        </select>
        <Button variant="ghost" disabled={busy} onClick={() => fileRef.current?.click()}>
          <IconUpload /> Upload CSV
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          hidden
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
        {datasetId && <Chip accent>ready</Chip>}
      </div>

      <Hairline />

      <SectionTitle>2 · Models under test</SectionTitle>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {models.map((m) => {
          const on = selected.includes(m.model_id);
          return (
            <button
              key={m.model_id}
              onClick={() => m.available && toggleModel(m.model_id)}
              disabled={!m.available}
              style={{
                ...modelChip,
                ...(on
                  ? { background: "var(--navy)", color: "#fff", border: "1px solid var(--navy)" }
                  : {}),
              }}
              title={m.description}
            >
              {m.name}
              {!m.available && " ·"}
            </button>
          );
        })}
      </div>
      <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
        <button className="btn btn-ghost" onClick={selectAll}>Select all</button>
        <button className="btn btn-ghost" onClick={() => setSelected([])}>Clear</button>
      </div>

      <Hairline />

      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <Button onClick={run} disabled={busy || !datasetId || selected.length === 0}>
          {busy ? "Running…" : `Run ${selected.length || ""} model${selected.length === 1 ? "" : "s"}`}
        </Button>
        {error && <span style={{ color: "#b04a3a", fontSize: 13 }}>{error}</span>}
      </div>

      {summary && (
        <>
          <Hairline />
          <SectionTitle>3 · Run summary</SectionTitle>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Mean confidence</th>
                <th>Loc MAE (km)</th>
                <th>Latency (ms/row)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary.models).map(([id, s]) => (
                <tr key={id}>
                  <td style={{ fontWeight: 600 }}>{id}</td>
                  <td className="mono">{fmt(s.accuracy, 4)}</td>
                  <td className="mono">{fmt(s.mean_confidence, 4)}</td>
                  <td className="mono">{fmt(s.loc_mae_km, 2)}</td>
                  <td className="mono">{fmt((s.latency || {}).mean_ms_per_row, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

const selectStyle = {
  padding: "8px 12px",
  borderRadius: 6,
  border: "1px solid var(--hairline)",
  background: "#fff",
  font: "inherit",
  color: "var(--ink)",
  minWidth: 260,
};

const modelChip = {
  padding: "6px 14px",
  borderRadius: 999,
  border: "1px solid var(--hairline)",
  background: "#fff",
  color: "var(--ink)",
  font: "inherit",
  fontSize: 13,
  cursor: "pointer",
};

function fmt(v, d) {
  return v == null ? "—" : Number(v).toFixed(d);
}
